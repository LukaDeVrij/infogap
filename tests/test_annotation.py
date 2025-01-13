import ipdb
import polars as pl
import pytest
from packages.annotate import get_candidate_annotations, annotate_frame

@pytest.fixture
# create an annotation dataframe
def annotation_frame():
    return pl.DataFrame({
        'fact': ['fact1', 'fact2', 'fact3'],
        'annotation_col_1': ['y', 'tbd', 'tbd'],
        'annotation_col_2': ['y', 'tbd', 'n']
    })

def test_get_candidate_annotations(annotation_frame):
    result_frame = get_candidate_annotations(annotation_frame, ['annotation_col_1', 'annotation_col_2'])
    # assert that the second and third rows are returned
    assert result_frame.shape == (2, 3)
    assert result_frame['fact'].to_list() == ['fact2', 'fact3']
    assert result_frame['annotation_col_1'].to_list() == ['tbd', 'tbd']

def test_annotate_frame(annotation_frame):
    result_frame = annotate_frame(annotation_frame, 2, ['annotation_col_1', 'annotation_col_2'],
        [lambda row: f"Is this fact good? {row['fact'][0]} (y/n)", lambda row: f"Is this fact bad? {row['fact'][0]} (y/n)"],
        [lambda x: x in ['y', 'n'], lambda x: x in ['y', 'n']],
        'test.csv',
        lambda x: 'y')
    assert result_frame['annotation_col_1'].to_list() == ['y', 'y', 'y'] # fill in the tbd values
    assert result_frame['annotation_col_2'].to_list() == ['y', 'y', 'n'] # fill in the tbd values

def test_annotation_frame_raise_keyboard_interrupt(annotation_frame):
    def raise_keyboard_interrupt(x):
        raise KeyboardInterrupt
    result_frame = annotate_frame(annotation_frame, 2, ['annotation_col_1', 'annotation_col_2'],
        [lambda row: f"Is this fact good? {row['fact'][0]} (y/n)", lambda row: f"Is this fact bad? {row['fact'][0]} (y/n)"],
        [lambda x: x in ['y', 'n'], lambda x: x in ['y', 'n']],
        'test.csv',
        raise_keyboard_interrupt)
    assert result_frame['annotation_col_1'].to_list() == ['y', 'tbd', 'tbd'] # nothing should have been updated
    assert result_frame['annotation_col_2'].to_list() == ['y', 'tbd', 'n'] # nothing should have been updated
    

def test_update_behaviour(annotation_frame):
    # convert tbd to null
    # then create a frame with updated annotations for the first two rows
    # then update the original frame with the updated annotations
    # then check that the original frame has been updated
    # annotation_frame['annotation_col_1'] = annotation_frame['annotation_col_1'].str.replace('tbd', None)
    # annotation_frame['annotation_col_2'] = annotation_frame['annotation_col_2'].str.replace('tbd', None)

    annotation_frame = annotation_frame.with_columns([
        pl.when(pl.col('annotation_col_1') == 'tbd').then(None).otherwise(pl.col('annotation_col_1')).keep_name(),
        pl.when(pl.col('annotation_col_2') == 'tbd').then(None).otherwise(pl.col('annotation_col_2')).keep_name()
    ])

    updated_frame = pl.DataFrame({
        'fact': ['fact1', 'fact2'],
        'annotation_col_1': ['y', 'y'],
        'annotation_col_2': ['y', 'y']
    })

    

    result_frame = annotation_frame.update(updated_frame, on='fact')
    # replace nones with tbds, using the with columns syntax.
    result_frame = result_frame.with_columns([
        pl.when(pl.col('annotation_col_1').is_null()).then(pl.lit('tbd')).otherwise(pl.col('annotation_col_1')).keep_name(),
        pl.when(pl.col('annotation_col_2').is_null()).then(pl.lit('tbd')).otherwise(pl.col('annotation_col_2')).keep_name()
    ])
    assert result_frame['annotation_col_1'].to_list() == ['y', 'y', 'tbd']
