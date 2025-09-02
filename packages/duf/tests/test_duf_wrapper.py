import pytest

import evo.data_converters.duf.common.deswik_types as dw


def test_cant_create_layer_with_duplicate_name():
    """
    This test is only valid for layers which existed before loading.

    This will not fail (as of this commit):
    >>> duf.NewLayer('new_layer')
    >>> duf.NewLayer('new_layer')
    """
    duf_file = r'data/polyline_attrs_boat.duf'

    duf = dw.Duf(duf_file)

    assert duf.LayerExists('0')

    with pytest.raises(dw.ArgumentException):
        duf.NewLayer('0')

    duf.NewLayer('new_layer')

    assert duf.LayerExists('new_layer')
    with pytest.raises(dw.ArgumentException):
        duf.NewLayer('new_layer')


def test_missing_file():

    with pytest.raises(dw.ArgumentException):
        duf = dw.Duf('not_a_real_file.duf')  # noqa: F841
