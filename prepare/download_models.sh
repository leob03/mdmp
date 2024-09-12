mkdir -p save
cd save

echo -e "Downloading pretrained model for HumanML3D dataset"
wget https://leobringer.com/wp-content/uploads/2024/09/preatrained_model.zip

echo -e "Unzipping preatrained_model.zip"
unzip preatrained_model.zip

#Oupsi aha
mv save/preatrained_model save/pretrained_model

echo -e "Cleaning preatrained_model.zip"
rm preatrained_model.zip

cd ../../

echo -e "Downloading done!"