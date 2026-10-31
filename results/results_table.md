| source | method | condition | level | accuracy_mean | accuracy_ci95 | macro_f1_mean | macro_f1_ci95 |
|---|---|---|---|---|---|---|---|
| synthetic | autoencoder+logreg | label_efficiency | 0.1 | 0.6666666666666666 | 0.0 | 0.5555555555555555 | 0.0 |
| synthetic | autoencoder+logreg | label_efficiency | 1.0 | 0.6666666666666666 | 0.0 | 0.5555555555555555 | 0.0 |
| synthetic | autoencoder+logreg | noise | 10.0 | 0.6190476190476191 | 0.11061622073262832 | 0.5954948571666838 | 0.12729273125990592 |
| synthetic | autoencoder+logreg | noise | 20.0 | 0.5619047619047619 | 0.026442334335217095 | 0.4929292929292929 | 0.019631430036752082 |
| synthetic | autoencoder+logreg | transfer | 1.0 | 0.6666666666666666 | 0.0 | 0.5555555555555555 | 0.0 |
| synthetic | pca+rf | label_efficiency | 0.1 | 0.38095238095238093 | 0.04180900157548021 | 0.2839682539682539 | 0.06324700770432831 |
| synthetic | pca+rf | label_efficiency | 1.0 | 0.43809523809523804 | 0.02644233433521713 | 0.3385391583533998 | 0.026631078207750954 |
| synthetic | pls-da | label_efficiency | 0.1 | 0.5047619047619047 | 0.06741498938010833 | 0.45800630169051226 | 0.08428092744428065 |
| synthetic | pls-da | label_efficiency | 1.0 | 0.38095238095238093 | 0.0 | 0.27777777777777773 | 0.0 |