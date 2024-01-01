# Scattering Optimization from Wires Structures

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository provides a framework for optimizing the electromagnetic performance of structures composed of wires. The
main component of the electromagnetic calculations is the [PyNEC](https://github.com/tmolteno/python-necpp) Python
package, which utilizes the Method of Moments (MoM) to calculate scattering, fields, and other electromagnetic
properties.

## Features

- Easily define the dimensions and geometries of the structures consisting of wires
- Use of optimization algorithms for finding the optimal wire configuration
- Calculation of scattering patterns from the optimized structure
- Visualization of the scattering patterns and the optimized structure and results export

## Installation

### Poetry

Initialize virtual environment

```shell
poetry shell
```

Then, install PyNEC package separately:

```shell
pip install pynec==1.7.3.4
```

And finally install other necessary requirements using:

```shell
poetry install
```

## Simple Usage

### Single-Seed Optimization

### Multi-Seed Optimization

For running multi-seed optimization, firstly, specify the `.yaml` config file like the
one [multi_seed_experiment.yaml](wirenec_optimization%2Fconfigs%2Fmulti_seed_experiment.yaml). The structure is the
following:

```yaml
parametrization_name: ...
parametrization_hyperparams: ...

optimization_hyperparams: ...

seeds_range: [ first_seed, last_seed, step ]
```

Then run the optimization:

```python
from omegaconf import OmegaConf

from wirenec_optimization.experiment import (
    MultiSeedOptimizationExperiment,
)

config = OmegaConf.load("configs/multi_seed_experiment.yaml")
experiment = MultiSeedOptimizationExperiment(config)
experiment.run()
```

After completing the optimization process, all results for individual seeds will be saved in
the [data/optimization](wirenec_optimization%2Fdata%2Foptimization) directory.

## Contributing

Contributions are welcome! If you find any bugs or want to suggest new features, or even more, use it in your own
research please create an issue or submit a pull request.

## Citing

If you would like to reference the framework in a publication, please use:

```
@article{grotov2022genetically,
  title={Genetically designed wire bundle superscatterers},
  author={Grotov, Konstantin and Vovchuk, Dmytro and Kosulnikov, Sergei and Gorbenko, Ilya and Shaposhnikov, Leon and Ladutenko, Konstantin and Belov, Pavel and Ginzburg, Pavel},
  journal={IEEE Transactions on Antennas and Propagation},
  volume={70},
  number={10},
  pages={9621--9629},
  year={2022},
  publisher={IEEE}
}
```

## Publications

1. Genetically Designed Wire Bundle Superscatterers. IEEE Transaction on Antennas and Propagation. Konstantin Grotov et
   al., 2022
2. Superradiant Scattering Limit for Arrays of Subwavelength Scatterers. Physical Review Applied. Anna Mikhailovskaya et
   al., 2022
3. Superradiant Broadband Magneto-electric Arrays Empowered by Meta-learning. ArXiv. Konstantin Grotov et al., 2023

## License

MIT License. See [LICENSE](https://github.com/yourusername/sos-wires/blob/main/LICENSE) for more information.