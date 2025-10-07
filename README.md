# pwlsplit

This package determines split points (indices) for piecewise linear data using second derivatives and refinement techniques.

## Features

- Identifies optimal split points in data series consists of piecewise linear segments
- Refines split locations for improved accuracy


## Method
- Provide a list of segments
- Smooth data for noise
- Compute second derivative to transform transition points into peaks and valleys
- Use numpy peak finding algorithm to find peaks indicies that conforms to preset
- Refine/optimize indicies for original data with noise

## Usage

Describe your data and call the provided functions to obtain split indices.

## Installation

```bash
pip install pwlsplit
```

## Example

```python
import pwlsplit

data = [ ... ]  # your piecewise linear data
splits = pwlsplit.find_splits(data)
print(splits)
```

## License

MIT