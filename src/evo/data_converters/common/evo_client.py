import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from uuid import UUID

import evo.logging
from evo.aio import AioTransport
from evo.common import ApiConnector, Environment, NoAuth
from evo.common.interfaces import ITransport
from evo.common.utils.cache import Cache
from evo.data_converters.common.exceptions import ConflictingConnectionDetailsError, MissingConnectionDetailsError
from evo.oauth import AuthorizationCodeAuthorizer, ClientCredentialsAuthorizer, OAuthScopes, OIDCConnector
from evo.objects import ObjectServiceClient
from evo.objects.utils.data import ObjectDataClient

if TYPE_CHECKING:
    from evo.notebooks import ServiceManagerWidget

logger = evo.logging.getLogger("data_converters")


@dataclass
class EvoWorkspaceMetadata:
    org_id: str = ""
    workspace_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    user_id: str = ""
    hub_url: str = ""
    redirect_url: str = "http://localhost:32369/auth/callback"
    oidc_issuer: str = ""
    cache_root: str = "./data/cache"

    def has_authentication_code_params(self) -> bool:
        return bool(self.client_id and self.hub_url and self.oidc_issuer and self.redirect_url)

    def has_client_credentials_params(self) -> bool:
        return bool(self.client_id and self.client_secret and self.oidc_issuer and self.user_id)


@dataclass
class EvoObjectMetadata:
    object_id: UUID
    version_id: Optional[str] = None


async def _authorization_code_authorizer(
    transport: ITransport, metadata: EvoWorkspaceMetadata
) -> AuthorizationCodeAuthorizer:
    authorizer = AuthorizationCodeAuthorizer(
        redirect_url=metadata.redirect_url,
        scopes=OAuthScopes.openid | OAuthScopes.evo_discovery | OAuthScopes.evo_object,
        oidc_connector=OIDCConnector(
            transport=transport,
            oidc_issuer=metadata.oidc_issuer,
            client_id=metadata.client_id,
        ),
    )
    await authorizer.login()

    return authorizer


async def client_credentials_authorizer(
    transport: ITransport, metadata: EvoWorkspaceMetadata
) -> ClientCredentialsAuthorizer:
    authorizer = ClientCredentialsAuthorizer(
        oidc_connector=OIDCConnector(
            transport=transport,
            oidc_issuer=metadata.oidc_issuer,
            client_id=metadata.client_id,
            client_secret=metadata.client_secret,
        ),
        scopes=OAuthScopes.openid
        | OAuthScopes.evo_discovery
        | OAuthScopes.evo_workspace
        | OAuthScopes.evo_object
        | OAuthScopes.evo_file,
    )
    await authorizer.authorize()

    return authorizer


def create_evo_object_service_and_data_client(
    evo_workspace_metadata: Optional[EvoWorkspaceMetadata] = None,
    service_manager_widget: Optional["ServiceManagerWidget"] = None,
) -> tuple[ObjectServiceClient, ObjectDataClient]:
    if evo_workspace_metadata and service_manager_widget:
        raise ConflictingConnectionDetailsError(
            "Please provide only one of EvoWorkspaceMetadata or ServiceManagerWidget."
        )
    elif evo_workspace_metadata:
        return create_service_and_data_client_from_metadata(evo_workspace_metadata)
    elif service_manager_widget:
        return create_service_and_data_client_from_manager(service_manager_widget)
    raise MissingConnectionDetailsError(
        "Missing one of EvoWorkspaceMetadata or ServiceManagerWidget needed to construct an ObjectServiceClient."
    )


def create_service_and_data_client_from_manager(
    service_manager_widget: "ServiceManagerWidget",
) -> tuple[ObjectServiceClient, ObjectDataClient]:
    logger.debug("Creating ObjectServiceClient from ServiceManagerWidget")
    environment = service_manager_widget.get_environment()
    connector = service_manager_widget.get_connector()
    service_client = ObjectServiceClient(environment, connector)
    data_client = service_client.get_data_client(service_manager_widget.cache)

    return service_client, data_client


def create_service_and_data_client_from_metadata(
    metadata: EvoWorkspaceMetadata,
) -> tuple[ObjectServiceClient, ObjectDataClient]:
    logger.debug(
        "Creating evo.objects.ObjectServiceClient and evo.objects.utils.data.ObjectDataClient with "
        f"EvoWorkspaceMetadata={metadata}"
    )

    cache = Cache(root=metadata.cache_root, mkdir=True)
    transport = AioTransport(user_agent="evo-data-converters")
    authorizer = NoAuth

    org_uuid = UUID(metadata.org_id) if metadata.org_id else metadata.org_id
    if metadata.has_client_credentials_params():
        authorizer = asyncio.run(client_credentials_authorizer(transport, metadata))
        hub_connector = ApiConnector(
            base_url=metadata.hub_url,
            transport=transport,
            authorizer=authorizer,
            additional_headers={"s2s-org-info": metadata.org_id, "s2s-user-info": metadata.user_id},
        )
    else:
        if metadata.has_authentication_code_params():
            authorizer = asyncio.run(_authorization_code_authorizer(transport, metadata))
        else:
            logger.debug("Skipping authentication due to missing required parameters.")

        hub_connector = ApiConnector(base_url=metadata.hub_url, transport=transport, authorizer=authorizer)

    workspace_uuid = UUID(metadata.workspace_id) if metadata.workspace_id else metadata.workspace_id

    environment = Environment(
        hub_url=metadata.hub_url,
        org_id=org_uuid,
        workspace_id=workspace_uuid,
    )
    service_client = ObjectServiceClient(environment, hub_connector)
    data_client = service_client.get_data_client(cache)

    return service_client, data_client
