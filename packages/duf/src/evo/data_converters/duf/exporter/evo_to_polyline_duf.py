import numpy
import pandas

from evo.data_converters.duf.fetch import FetchedPolyline


def polyline_to_duf(fetched_polyline: FetchedPolyline, duf):

    # TODO Need to handle duplicate names
    new_layer = duf.NewLayer(fetched_polyline.name)

    evo_to_dw_type_lookup = {
        'string': 'String',
        'scalar': 'Double',
        'category': 'String',
        'date_time': 'DateTime',
        'integer': 'Integer',
    }

    def convert_from_maybe_np_type(maybe_np):
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

    dw_attributes = []
    if len(fetched_polyline.attributes) > 0:
        for attrs in fetched_polyline.attributes:
            dw_attributes.append(new_layer.AddAttribute(attrs['name'], evo_to_dw_type_lookup[attrs['type']]))

    for i, path in enumerate(fetched_polyline.paths):
        new_polyline = duf.NewPolyline(new_layer)
        new_polyline.SetVertices3D(path.flatten())

        for dw_attr, evo_attrs in zip(dw_attributes, fetched_polyline.attributes):
            # TODO check types of these values and make sure it won't cause a problem with the bindings
            value = evo_attrs['values'][i]
            converted = convert_from_maybe_np_type(value)
            if converted is not None:
                new_polyline.SetAttribute(dw_attr, converted)
