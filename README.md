# Italian Deep Ensemble Model 
This is the model used on Task A: Hate Speech Detection at Evalita 2020.

This project uses FreeLing 4.0 Tool. To archive a successful performance, you most install FreeLing with the python API and run:
    cp $FREELINGDIR/share/freeling/APIs/python3/_pyfreeling.so  code/_pyfreeling.so
    cp $FREELINGDIR/share/freeling/APIs/python3/_pyfreeling.py  code/pyfreeling.py
where $FREELINGDIR is the installation path of Freeling (deafult: /usr/local)

The data is palced inside a data folder:
    mkdir data
    cp DIRDATA/haspeede2_dev_taskAB.tsv data/haspeede2_dev_taskAB.tsv
    cp DIRDATA/haspeede2-test_taskAB-tweet.tsv data/haspeede2-test_taskAB-tweet.tsv
    cp DIRDATA/haspeede2-test_taskAB-news.tsv data/haspeede2-test_taskAB-news.tsv
    cp DIRDATAOLD/training_set_senticpolc16.tsv data/training_set_senticpolc16.tsv
where DIRDATA is the data location for haspeede2 dataset and DIRDATAOLD for sentipolc_2016 dataset.

The embedding used for now is not available.

* Urls:
    Freeling: ...
    Evalita_2020: ...
    Senticpolc_2016: ...