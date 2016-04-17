mkdir sample
mkdir temp
mkdir sep_files
wget "http://www.hark.jp/wiki.cgi?page=SupportedHardware&file=microcone%5Frectf%2Ezip&action=ATTACH" -O sample/microcone_rectf.zip
wget "http://www.hark.jp/wiki.cgi?page=SupportedHardware&file=microcone%5Fgeotf%2Ezip&action=ATTACH" -O sample/microcone_geotf.zip
wget "http://www.hark.jp/wiki.cgi?page=SupportedHardware&file=tamago%5Frectf%2Ezip&action=ATTACH"  -O sample/tamago_rectf.zip


git clone https://github.com/naegawa/HARK_TF_Parser.git

