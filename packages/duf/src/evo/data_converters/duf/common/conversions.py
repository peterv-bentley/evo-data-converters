
import numpy
import pandas

from evo.data_converters.duf.common.types import AttributedEvoData, FetchedTriangleMesh, FetchedLines, EvoAttributes

EVO_TO_DW_TYPE_CONVERSION = {
    'string': 'String',
    'scalar': 'Double',
    'category': 'String',
    'date_time': 'DateTime',
    'integer': 'Integer',
}


def np_to_dw(maybe_np):
    if isinstance(maybe_np, str) or numpy.issubdtype(maybe_np, numpy.str_):
        return str(maybe_np)
    elif isinstance(maybe_np, float) or numpy.issubdtype(maybe_np, numpy.floating):
        f = float(maybe_np)
        if numpy.isnan(f):
            # TODO review returning an empty string. This is how it appeared to me when importing Deswik entities
            #  that were missing attributes.
            return ''
        else:
            return f
    elif numpy.issubdtype(maybe_np, numpy.datetime64):
        # TODO Check the expected datetime format in Deswik.Cad
        return pandas.to_datetime(maybe_np).strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(maybe_np, int) or numpy.issubdtype(maybe_np, numpy.integer):
        return int(maybe_np)
    else:
        raise NotImplementedError(f'Unhandled type {type(maybe_np)}')


class Layer:
    def __init__(self, layer, dw_attributes):
        self.layer = layer
        self._attributes = dw_attributes

    @staticmethod
    def with_attributes(duf, evo_data: AttributedEvoData):
        # TODO Need to handle duplicate names
        new_layer = duf.NewLayer(evo_data.name)
        dw_attributes = {
            attr.name: new_layer.AddAttribute(attr.name, EVO_TO_DW_TYPE_CONVERSION[attr.type])
            for attr in evo_data.attributes
        }
        return Layer(new_layer, dw_attributes)

    def __getitem__(self, item: str):
        return self._attributes[item]

    def set_attributes_to_entity(self, dw_entity, evo_attributes: list[EvoAttributes], i: int):
        for attr in evo_attributes:
            value = attr.values[i]
            converted = np_to_dw(value)
            dw_attr = self[attr.name]
            dw_entity.SetAttribute(dw_attr, converted)


class EvoDufWriter:
    def __init__(self, duf):
        self._duf = duf

    def write_mesh_triangles(self, mesh_triangles: FetchedTriangleMesh):
        new_layer = Layer.with_attributes(self._duf, mesh_triangles)

        flattened_vertices = mesh_triangles.vertices.flatten()

        for i, part in enumerate(mesh_triangles.parts):
            new_polyface = self._duf.NewPolyface(new_layer.layer)
            # TODO This is an annoying amount of overhead to pay. Pythonnet converts automatically for double arrays
            #  (and maybe even avoids a copy), but it work for numpy integer arrays. Consider handling the numpy objects
            #  explicitly in the Duf wrapper library.
            flattened_triangles = part.flatten().tolist()
            new_polyface.SetVertices3D(flattened_vertices, flattened_triangles)

            new_layer.set_attributes_to_entity(new_polyface, mesh_triangles.attributes, i)

    def write_lines(self, lines: FetchedLines):
        new_layer = Layer.with_attributes(self._duf, lines)

        for i, path in enumerate(lines.paths):
            new_polyline = self._duf.NewPolyline(new_layer.layer)
            new_polyline.SetVertices3D(path.flatten())

            new_layer.set_attributes_to_entity(new_polyline, lines.attributes, i)
