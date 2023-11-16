# AMMASI
Areal embedding based Masked Multihead Attention-based Spatial Interpolation for House Price Prediction

![table1-performance-comparison](figs/fig1-proposed-model.jpg)

Improved on top of [ASI](https://github.com/darniton/ASI/) [Viana2021].

# Dataset

![table1-performance-comparison](figs/fig2-areal-embedding.jpg)

Original data is uploaded in [ASI](https://github.com/darniton/ASI/).


# Experimental Result of AMMASI

![table1-performance-comparison](figs/table1-performance-comparison.jpg)




## Results of ASI
* ASI with HA is the result from [Viana2021]
* ASI with HA+P is the result recorded in notebooks/{dname}/Training-norm.ipynb

※ The file name 'norm' has nothing to do with poi. Sorry for confusion.

In asi_norm/input_dataset.py (line 49): 
```
    def __call__(self):

        assert isinstance(self.id_dataset, object)

        data = np.load(PATH + '/datasets/'+ self.id_dataset + '/data_poi.npz', allow_pickle=True)

```

## Results of AMMASI

* Full results are reported in *mycode/prediction/make_table-AMMASI.ipynb*
* You can also check *mycode/test_logs* for the full experiment logs.
* You can also refer to ablation study (Fig.5).

![table1-performance-comparison](figs/fig5-sigma-study.jpg)


## Parameter and Ablation Test


![table1-performance-comparison](figs/fig3-poi-proximity.jpg)

![table1-performance-comparison](figs/fig4-poi-coefficient.jpg)





# References

[Viana2021] Viana, Darniton, and Luciano Barbosa. "Attention-based spatial interpolation for house price prediction." Proceedings of the 29th International Conference on Advances in Geographic Information Systems. 2021.