# HetMNA

Source code and datasets for 2021 paper: [***？？***](https://.pdf).

## Datasets

> Please first download the datasets [here](??) and extract them into `data/` directory.

Initial datasets WN31-15K, DBP15K and DWY100K are from [OpenEA](https://github:com/nju-websoft/OpenEA) and [JAPE](https://github.com/nju-websoft/JAPE).

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
* Scipy
* Numpy


## Running

For example, to run HetMNA on DBP15K (ZH-EN), use the following script:
```
python3 main.py --dataset DBP15k --lang zh_en
```


> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any difficulty or question in running code and reproducing expriment results, please email to caiws@m.scnu.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

*Weishan Cai, Yizhao Wang, Shun Mao, Jieyu Zhan and Yuncheng Jiang. ??. 2021.*
