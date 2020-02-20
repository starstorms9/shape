# 3D Generative Design
Tyler Habowski Insight AI Project | Session AI.SV.2020A

### Project Goal:  
Generate 3D models from text input to increase the speed and efficacy of the design process.

# Streamlit App Manual
See the demo live at [datanexus.xyz](https://www.datanexus.xyz)
            
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

### Example pre-rendered gifs below:
Interpolating between various swivel chairs:

![swivel chairs gif](https://github.com/starstorms9/shape/blob/master/media/swivelchairs.gif "")

Interpolating between various random sofas:  

![couches gif](https://github.com/starstorms9/shape/blob/master/media/couches.gif "")

