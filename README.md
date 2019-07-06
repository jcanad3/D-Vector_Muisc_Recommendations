# D-Vector_Music_Recommendations
*A repository for unsupervised music recommendation using d-vectors.*

## Introduction
Recently, RNNs have been trained to approximate small intervals of audio signals with finite
length embeddings. These embeddings are known as d-vectors. Thus, by using d-vectors one
can calculate an embedding for an audio clip of indeterminate length by averaging the extracted
d-vectors from each window. A popular use case of this method of summarizing audio
is the task of speaker verification. In this project, however, d-vectors are used to
represent songs from the Free Muisc Archive dataset. Similar to the speaker verification task,
the Generalized End-to-End (Wang et al.) loss function is used to train the d-vector extractor. An open 
source PyTorch implementation, found here https://github.com/HarryVolek/PyTorch_Speaker_Verification,
was modified to create an average embedding for each song. These embeddings are then
visualized with the manifold learning technique t-SNE to reveal any shared latent characteristics
of the embeddings (see tsne_vis.ipynb). After calculating the embeddings for a subset of the songs, the unsupervised
nearest neighbors algorithm, provided by scikit-learn, with a cosine distance metric, is used to determine
song recommendations (see nearest_songs.ipynb).

## Models
Two different models were trained to produce d-vectors: a song based model and a genre based model.

#### Song only D-Vector Model
A training batch for this model consisted of random samples from Log Mel-Spectrograms for five different songs.
The model tries to maximize the cosine distance between its own centroid of embeddings and the centroid of embeddings
for the nearest song. This process is further illustrated in the paper "Generalized End-to-End Loss for Speaker
Verification" published by Wang et al.

#### Genre D-Vector Model
A training batch for the genre based model consisted of random Log Mel-Spectrogram samples from songs within the same
genre. Five gernes were in every batch, with ten samples per genre. GE2E Loss is still applied in this scenario,
although the model now learns to maximized the distance between genres instead of just songs. 

## Results
It is difficult to get an accurate measurement for song recommendations without a large body
of volunteers to label the validity of suggested songs. However, the few samples that were inspected
by ear were found to be quite relevant. Recommended songs often fall within the same genre and intuitively
sound quite similar. The d-vector model/nearest neighbors combo was also able to detect duplicate entries
in the Free Musich Archive dataset, i.e. identical songs with different ids. This suggests that this 
technology could be used to determine whether a new song violates copyright issues by being to similar
to another previously released song. A threshold value could be set on the cosine distance/similarity between songs,
which when exceeded, indicates that the two songs are too similar to be considered unique. 

Another use, as well as the original intended use of this project, is to recommned songs that sound similar to a 
song that a user admires. One could create a profile for each user by averaging the d-vectors of all of 
their favorited songs, then perform nearest neighbors search on that profile embedding. One could also 
create an average embedding across all songs that user deems unfavorable, then perform farthest neighbors 
search to recommend songs.
