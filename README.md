# pwlsplit

This package is used for finding split points (indices) for piecewise linear data with noise.

Although `pwlf` module exists, it is difficult to use for large problems, as the optimization
algorighms have significant difficulties. Here, local minimals are insufficient by definition.
Global objective function surface has too many saddle points and minima to be solvable in a
reasonable amount of time.

## Strategy

The goal is to take advantage of the fact that the second derivative of piecewise linear curves
are delta functions at the transition points. This make obvious peaks and valleys. Signal
processing has fairly advanced and algorithms for finding these peaks.

Challenges, noise. Local changes in second derivatives are significantly greater than those
due to the transition points. Rather, a transition points is typically observed as a range with
more successive positive increases rather than a distinct peaks. The most obvious strategies are
1) using statisitics to find changes not explained by noise, or
2) **smooth the data.**

New challenge here is that the peaks and valleys are now finite in height, and not known unless
given by the users. However, if minimal info is supplied. We can have a fairly efficient strategy.

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
pip install .
```

## Example

see pwlsplit/bogoni

## License

MIT