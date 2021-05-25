# PhyPy 802.11ad components 

Python 802.11ad (WiGig) component definitions, including:
- Constellation mapper
- Constellation demapper (optimal, sub-optimal, and decision threshold algorithms)
- LDPC encoder
- LDPC decoder (min-sum and sum-product algorithms)
- Scrambler / descrambler
- AWGN channel (noise based on desired EbNo input value)

Numba JIT was used where possible to reduce execution times.

## Install

### Basic
`pip3 install .`

## Tests

Unit tests are included to verify the conformity of individual components. 

### Run

The following commands will attempt to run the package contained withing this directory (not an installed instance). 

#### Command line

```python3 -m unittest discover -s Tests/ -p '*TestCase.py'```

#### Pycharm

- Go to ```Run > Edit Configurations```
- Click *Add new configuration (+)*
- Navigate and click ```Python tests > Unittests```
- Select target folder (script path): ```Tests``` or any of its subdirectories
- Enter *Pattern* ```*TestCase.py```
- Run

For more configuration options, view [PyCharm docs on unit testing](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python-unit-test.html).


### Troubleshooting

##### Unitests doesn't execute any tests (output is OK)
Go to ```Run > Edit Configurations > Python tests > [test name]``` and add the *Pattern* ```*TestCase.py```

##### Another test framework is being run
Go to ```CTRL +  ALT + S > Tools > Python Integrated Tools``` and select ```Unittests``` as the *Default test runner*

##### AssertionError: Path must be within the project
In ```Run > Edit Configurations > Environment``` check the *Working directory* is the project base.

### Designing tests
- Add an empty ```__init__.py``` file to each directory that contains a TestCase on the same or on a lower level.
- Each file containing a TestCase needs to end with ```TestCase.py```
