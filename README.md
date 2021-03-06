# Music Generation

The FMA dataset can be found [here](https://github.com/mdeff/fma)
GTZan dataset can be found [here](https://marsyasweb.appspot.com/download/data_sets/)

## Setup Instructions
```
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
cd code
```

## Execution Instructions

For LSTM:
```python
# enter into LSTM folder
cd LSTM

# generate input sequence
python3 mfcc_to_wav.py --input /path/to/input_songs

# find right hyperparameters for the LSTM model
python3 tune_hyperparameters.py

# train model with optimal parameters
python3 train_lstm.py

# generate new music sequences
# Output file format must be .wav
python3 generate.py --output /path/to/new_song_sequence.wav
```

For VAE:
```python
# enter into VAE folder
cd VAE

# generate the input representation
python3 wav_to_mfcc.py

# run the model
python3 vae_try2.py
```

# Contribution Guidelines
1. Fork this repository
2. Clone your fork
3. Set this repository as the upstream
```
git remote add upstream https://github.com/hardiksurana/music-generation
```
4. Switch to a new branch
```
git checkout -b <NEW_BRANCH_NAME>
```
4. Make changes to that branch
5. Commit your changes
```
git commit -am "<COMMIT_MSG>"
```
6. Rebase with this repository's master branch to account for new changes
```
git fetch upstream
```

After fixing merge conflicts (if any)

```
git rebase upstream/master
```
7. Push the changes to your fork
```
git push origin <NEW_BRANCH_NAME>
```
8. Send a Pull Request to this repository