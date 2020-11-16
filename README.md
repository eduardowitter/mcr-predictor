# mcr-predictor


## Source tree

Each research question has its directory with the associated scripts, and the shared
code is in `lib`.

```
mcr-predictor$ tree
.
├── lib
│   └── dataset.py
├── LICENSE
├── raw_dataset.csv
├── README.md
├── rq1
│   ├── rq1_feedback_count.py
│   └── rq1_is_reviewer.py
├── rq2
│   ├── rq2_feedback_count.py
│   ├── rq2_feedback_count_without_f2_f3.py
│   └── rq2_is_reviewer.py
├── rq3
│   ├── rq3_feedback_count.py
│   └── rq3_is_reviewer.py

```

## After cloning this repository

- Extract `dataset.tar.gz`:
```
tar -zxvf dataset.tar.gz
```
Then, run any of the scripts listed above
```
python3 -u <script name>
```

## Dataset

Due to a confidentiality agreement, only data from 2018 is disclosed.
