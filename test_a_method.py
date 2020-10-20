def test_dummy():
    assert dummy(1,100) ==101

@pytest.mark.parametrize(
    "input,expected", 
    [
        ("3+5", 8),
        ("2+4", 6),
        pytest.param("6*9", 100, marks=pytest.mark.xfail(reason="wrong answer")),
    ]
)

def test_dummy_parametrize(input, expected):
    assert dummy_parametrize(input) == expected

d1 = {1: 'apple', 2: 'ball'}
d2 = {1: 'apple', 2: 'ball'}
d3 = {1: 'apple', 2: 'zebra'}

@pytest.mark.parametrize(
    "input, output", 
    [
        (100, 100),
        (d1, d2),
        pytest.param(d1, d3, marks=pytest.mark.xfail(reason="wrong answer")),
    ]
)

def test_dummy_compare_dict(input, output):
    d_out = dummy_compare_dict(input)
    
    d_out_str = json.dumps(d_out)
    output_str = json.dumps(output)

    assert d_out_str == output_str
