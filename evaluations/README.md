# Setup

## Marc-ja
```
wget https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_JP_v1_00.tsv.gz
wget https://raw.githubusercontent.com/yahoojapan/JGLUE/main/preprocess/marc-ja/scripts/marc-ja.py
mkdir datasets
wget https://raw.githubusercontent.com/yahoojapan/JGLUE/main/preprocess/marc-ja/data/filter_review_id_list/valid.txt -O marc-ja_filter_review_id_list_valid.txt
wget https://raw.githubusercontent.com/yahoojapan/JGLUE/main/preprocess/marc-ja/data/label_conv_review_id_list/valid.txt -O marc-ja_label_conv_review_id_list_valid.txt
gzip -dc amazon_reviews_multilingual_JP_v1_00.tsv.gz | python marc-ja.py --positive-negative --output-dir datasets/marc_ja-v1.1 --max-char-length 500 --filter-review-id-list-valid marc-ja_filter_review_id_list_valid.txt --label-conv-review-id-list-valid marc-ja_label_conv_review_id_list_valid.txt
head datasets/marc_ja-v1.1/train-v1.0.json
head datasets/marc_ja-v1.1/valid-v1.0.json
```
