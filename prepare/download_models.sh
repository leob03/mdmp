mkdir -p save
cd save

echo -e "Downloading pretrained model for HumanML3D dataset"
wget https://leobringer.com/wp-content/uploads/2024/09/preatrained_model.zip

echo -e "Unzipping preatrained_model.zip"
unzip preatrained_model.zip

echo -e "Renaming folder pretrained_model to mdmp_pretrained"
mv preatrained_model mdmp_pretrained

echo -e "Cleaning preatrained_model.zip"
rm preatrained_model.zip

cd ../../

echo -e "Downloading done!"