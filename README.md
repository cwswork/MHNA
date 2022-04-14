# MHNA

Source code and datasets for 2022 paper: [***Multi-Heterogeneous Neighborhood-Aware for Knowledge Graphs Alignment, IPM, 2022***]

## Datasets

> Please first download the datasets [here](https://www.jianguoyun.com/p/DY8iIAsQ2t_lCBjK3oUEIAA) and extract them into `datasets/` directory.

Initial datasets WN31-15K and DBP-15K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

Initial datasets DWY100K is from  [BootEA](https://github.com/nju-websoft/BootEA) and [MultiKE](https://github.com/nju-websoft/MultiKE).

Take the dataset EN_DE(V1) as an example, the folder "pre " contains:
* kg1_ent_dict: ids for entities in source KG;
* kg2_ent_dict: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* rel_triples_id: relation triples encoded by ids;
* attr_triples_id: attribute triples encoded by ids;
* kgs_num: statistics of the number of entities, relations, attributes, and attribute values;
* value_embedding.out: the input entity feature matrix initialized by word vectors;
* entity_embedding.out: the input attribute value feature matrix initialized by word vectors;


## Environment

* Python>=3.7
* pytorch>=1.7.0
* tensorboardX>=2.1.0
* Numpy
* json


## Running

To run MHNA model on WN31-15K and DBP-15K, use the following script:
```
python3 align_exc_15K.py
```
To run MHNA model DWY100K, use the following script:
```
python3 align_exc_DWY100K.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to cwswork@qq.com.

## Citation

If you use this model or code, please cite it as follows:

*Weishan Cai, Yizhao Wang, Shun Mao, Jieyu Zhan and Yuncheng Jiang. Multi-heterogeneous neighborhood-aware for knowledge graphs alignment. Information Processing & Management, 2022, 59(1): 102790.
