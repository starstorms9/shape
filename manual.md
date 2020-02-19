# Streamlit App Manual
This is my streamlit app for my Insight AI.SV.2020A project.
            
## Available Tabs:            
- ### Text to shape generator
- ### Latent vector exploration
- ### Shape interpolation

## Text to Shape Generator
This tab allows you to input a description and the generator will make a model based on that description.
The 3D plotly viewer generally works much faster in Firefox compared to chrome so use that if chrome is being slow.

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

(show some descriptions)
        

## Latent Vector Exploration
This tab shows the plot of the shape embedding vectors reduced from the full model dimensionality of 128 dimensions
down to 2 so they can be viewed easily. The method for dimensionality reduction was TSNE.

#### In the exploration tab, there are several sidebar options:
  - Color data
    - This selector box sets what determines the color of the dots. (the class selections are particularly interesting!)
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
'media/tsne_small.png'

## Shape Interpolation
This tab is just for fun and is intended to show how well the model can interpolate between various 
object models. Note that this runs the model many times and as such can be quite slow online. You may need to hit 'stop' 
and then 'rerun' from th menu in the upper right corner to make it behave properly.

To generate these plots, the algorithm finds the nearest K shape embedding vectors
(K set by the variety parameter in the sidebar) and randomly picks one of them.
Then it interpolates between the current vector and the random new vector
and at every interpolated point it generates a new model from the interpolated latent space vector.
Then it repeats indefinitely finding new vectors as it goes.

#### In this tab there are 2 sidebar options:
  - Starting shape
  - This sets the starting category for the algorithm but it will likely wander off into other categories
  after a bit
  - Variety parameter
  - This determines the diversity of the models.

### Example pre-rendered gif below:
'media/couches.gif'
