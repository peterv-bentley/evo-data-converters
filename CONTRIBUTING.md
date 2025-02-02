# Contributing

We welcome all types of contributions.

Found a bug? Please create an [issue](https://github.com/seequent/evo-samples/issues).

Want to contribute by creating a pull request? Great!
[Fork this repository](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/working-with-forks) to get started.

## Pull Requests

Before creating a pull request, make sure your changes address a specific issue.
Do a search to see if there are any existing issues that are still open.
If you don't find one, you can create one.

To enable us to quickly review and accept your pull requests, always create one pull request per issue.
Never merge multiple requests in one unless they have the same root cause.
Be sure to follow best practices and keep code changes as small as possible.
Avoid pure formatting changes or random "fixes" that are unrelated to the linked issue.

## Building a Release

### Prerequisites

- GitHub access with appropriate permissions to create releases and run workflows
- Ensure all tests are passing in the main branch
- Check there is a draft release named "Upcoming release" at https://github.com/seequent/evo-samples-internal/releases with a list of changes since last release

### Release Process

Navigate to the Release workflow at https://github.com/seequent/evo-samples-internal/actions/workflows/release.yml and click "Run workflow"

The workflow will automatically:

- Bump the current version of the project according to your selection of a major, minor, patch release
- Tag the new release and push to the repository
- Build and publish a new version of the wheel
- Publish the draft release

### Version Bumping Guidelines

Choose the appropriate version bump based on the changes included:

- Major (1.0.0 → 2.0.0): Breaking changes that are not backward compatible
- Minor (1.0.0 → 1.1.0): New features that maintain backward compatibility
- Patch (1.0.0 → 1.0.1): Bug fixes and minor improvements
