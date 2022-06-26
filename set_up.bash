# see Antimicrobial_Peptide_QSAR.ipynb in https://github.com/dataprofessor/peptide-ml

wget https://github.com/raghavagps/Pfeature/raw/master/PyLib/Pfeature.zip
unzip Pfeature.zip
cd Pfeature

python setup.py install #NB I changed: packagedata={'Pfeature': ['/*']} to packagedata={'Pfeature': ['.']} 

conda install -c bioconda cd-hit -y

#Load peptide dataset
wget https://raw.githubusercontent.com/dataprofessor/AMP/main/train_po.fasta
wget https://raw.githubusercontent.com/dataprofessor/AMP/main/train_ne.fasta

#Remove redundant sequences using CD-HIT
cd-hit -i train_po.fasta -o train_po_cdhit.txt -c 0.99
cd-hit -i train_ne.fasta -o train_ne_cdhit.txt -c 0.99














