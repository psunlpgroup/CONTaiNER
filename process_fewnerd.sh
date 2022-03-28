unzip episode_data.zip &&
python misc/cvt_conll.py &&
python misc/cvt_to_lowercase.py &&
# cleanup
rm -rf episode*