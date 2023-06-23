This repository contains the code used in the article "Radial-angular coupling in self phase modulation with structured light" by Gil de Oliveira et. al.

The `projections.jl` file contains the code to generate the most of the plots in the article. It consists mostly of plotting functions, because the heavy numerical lifting comes from [StructuredLight.jl](https://github.com/marcsgil/StructuredLight.jl). The insets are calculated in `insets.jl`. The file `non_linear_power.jl` contains the data from Table 1.

In order to use this code, clone the repository, open a Julia REPL, press `]` to enter the `Pkg` mode, and then run `instantiate` to download the required dependencies.