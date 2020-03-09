# 3D Generative Design
Tyler Habowski Insight AI Project | Session AI.SV.2020A

### Project Goal:  
Generate 3D models from text input to increase the speed and efficacy of the design process.

# Repo Usage
To run the code on this repo follow these steps:
 1. Download all data and update configuration file according to download locations in **configs.py**.
 
      a. IMPORTANT NOTE: Downloading the ShapeNet / PartNet databases requires authorization from the organizers, usually takes ~3-4 business days.
      
      b. Once approved, download ShapeNetCore database called 'Archive of ShapeNetCore v2 release' from [here](http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip).
      
      c. PartNet also requires filling out an additional and separate form to download. That process can be started [here](https://www.shapenet.org/download/parts).
      
      d. Note that not all of the data from these databases is utilized and thus only parts of the archives need to be unzipped to save signficant time and memory. Only the .solid.binvox files are used from ShapeNetCore and only the .json files are used from PartNet to generate the descriptions. However, the programs assume they are in the same relative folder structure.
      
 2. Train shape encoder model using **vae.py** file. Confirm model performance using provided visualization methods.
 3. Gather data and generate descriptions for objects.
 
      a. Run through **partnetmeta.py** which gathers information from the PartNet database.
      
      b. Then run through **descriptor.py** which uses that output in order to generate the randomized object descriptions.
      
 4. Train text encoder model using **text2shape.py**. Confirm model performance using provided visualization methods.
 5. Create TSNE plots using **tsne.py** which generates a pandas csv file with relevant info for use in the streamlit app.
 6. Run the **streamlit_app.py** file with Streamlit.
 
      a. Navigate to the folder with the file in a terminal.
      
      b. Run the command "streamlit run streamlit_app.py".
      
The following files are used by the main programs:
- **utils.py** Contains many commonly useful algorithms for a variety of tasks.
- **textspacy.py** Class that contains the text encoder model.
- **cvae.py** Class that contains the shape autoencoder model.
- **logger.py** Class for easily organizing training information like logs, model checkpoints, plots, and configuration info.
- **easy_tf2_log.py** Easy tensorboard logging file from [here](https://github.com/mrahtz/easy-tf-log) modified for use with tensorflow 2.0

# Streamlit App Manual
See the demo live at [datanexus.xyz](http://datanexus.xyz) (works best with Firefox or Safari, not Chrome!)
            
## Available Tabs:            
- ### Text to shape generator
- ### Latent vector exploration
- ### Shape interpolation

## Text to Shape Generator
This tab allows you to input a description and the generator will make a model based on that description.
The 3D plotly viewer generally works much faster in Firefox compared to chrome so use that if chrome is being slow.

The bottom of this tab shows similar descriptions to the input description. Use these samples to see new designs and learn how the model interprets the text.

#### Models were trained on these object classes _(number of train examples)_:
- Table    (8436)
- Chair    (6778)
- Lamp     (2318)
- Faucet   (744)
- Clock    (651)
- Bottle   (498)
- Vase     (485)
- Laptop   (460)
- Bed      (233)
- Mug      (214)
- Bowl     (186)        

## Latent Vector Exploration
This tab shows the plot of the shape embedding vectors reduced from the full model dimensionality of 128 dimensions
down to 2 so they can be viewed easily. The method for dimensionality reduction was TSNE.

#### In the exploration tab, there are several sidebar options:
  - Color data
    - This selector box sets what determines the color of the dots (the class selections are particularly interesting!)
  - Plot dot size
    - This sets the dot size. Helpful when zooming in on a region.
  - Model IDs             
    - This allows for putting in multiple model IDs to see how they're connected on the graph.
  - **Anno IDs to view**
    - From the hover text on the TSNE plot points you can see the 'Anno ID' (annotation ID) and enter it into this box to see a render of the object and 1 of its generated descriptions.
    - Multiple IDs can be entered and separated by commas.
    - The renders can be viewed in the sidebar or in the main area below the TSNE graph.

Additionally, using the plotly interface you can **double click** on a category in the legend to show only that
category of dots. Or **click once** to toggle showing that category. You can also zoom in on specific regions to
see local clustering in which case it may be useful to increase the plot dot size.

The shape embeddings are very well clustered according to differt shape classes but also to sub categories
inside those classes. By playing with the color data, it can be seen that the clusters are also organized very strongly
by specific attributes about the object such as is it's overall width, length, or height.

### TSNE map showing different colors for the different shape classes:            
![tsne small map](https://github.com/starstorms9/shape/blob/master/media/tsne_small.png "")

## Shape Interpolation
This tab is just for fun and is intended to show how well the model can interpolate between various 
object models. Note that this runs the model many times and as such can be quite slow online. You may need to hit 'stop' 
and then 'rerun' from th menu in the upper right corner to make it behave properly.

To generate these plots, the algorithm finds the nearest K shape embedding vectors
(K set by the variety parameter in the sidebar) and randomly picks one of them.
Then it interpolates between the current vector and the random new vector
and at every interpolated point it generates a new model from the interpolated latent space vector.
Then it repeats to find new vectors.

#### In this tab there are 2 sidebar options:
  - Starting shape
    - This sets the starting category for the algorithm but it will likely wander off into other categories                after a bit
  - Variety parameter
    - This determines the diversity of the models by setting how many local vectors to choose from.

### Results Overview:
Selected results from the streamlit app:
![results.png](https://github.com/starstorms9/shape/blob/master/media/results.png "")

#### Shape Encoder Interpolations:
Interpolating between various swivel chairs:

![swivel chairs gif](https://github.com/starstorms9/shape/blob/master/media/swivelchairs.gif "")

Interpolating between various random sofas:  

![couches gif](https://github.com/starstorms9/shape/blob/master/media/couches.gif "")

(many more gifs available in the media/ folder of this repo)
