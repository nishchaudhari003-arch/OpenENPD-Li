# OpenENPD-Li
OpenENPD-Li is an open-source Python toolkit for extended Nernst–Planck–Donnan modeling of membrane-based lithium extraction from multicomponent brines. The project focuses on transparent, reproducible simulation workflows for ion transport through nanofiltration membranes, with an initial emphasis on Li⁺/Mg²⁺ separation from salt-lake-brine-inspired systems.

Project status:
This repository is under active development.
Current benchmark case:
1. Foo et al. 2023, Environmental Science & Technology
2. DOI: 10.1021/acs.est.2c08584
3. System: simplified Li⁺–Mg²⁺–Cl⁻ brine
4. Membrane: NF270
5. Condition: pH ≈ 7
6. Target output: Li⁺ and Mg²⁺ species rejection versus water flux

Repository structure

OpenENPD-Li/ 
├── data/ 
│ └── foo2023_lmc_ph7_full.csv 
├── notebooks/ 
│ └── 01_foo2023_lmc_ph7_data_check.ipynb 
├── openenpd/ 
│ └── __init__.py 
├── tests/ 
│ └── .gitkeep 
├── README.md 
└── requirements.txt

Initial benchmark
The first benchmark reproduces the experimental Li⁺ and Mg²⁺ rejection trends for the simplified LM-C brine case from Foo et al. 2023. The benchmark dataset includes pressure, water flux, pH, Li⁺ rejection, Mg²⁺ rejection, feed composition, and NF270 membrane parameters.

Planned features
1. Extended Nernst–Planck ion-transport solver
2. Donnan partitioning at membrane interfaces
3. Multicomponent electroneutrality handling
4. Lithium/magnesium selectivity prediction
5. Published-case validation notebooks
6. Parameter-estimation workflows
7. Uncertainty and identifiability analysis

Installation
pip install -r requirements.txt

Citation

The first validation dataset is based on:

Foo, Z. H.; Rehman, D.; Bouma, A. T.; Monsalvo, S.; Lienhard, J. H.
Lithium Concentration from Salt-Lake Brine by Donnan-Enhanced Nanofiltration.
Environmental Science & Technology, 2023, 57, 6320–6330.
DOI: 10.1021/acs.est.2c08584
