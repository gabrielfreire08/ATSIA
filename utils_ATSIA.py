######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################## PARA ATSIA###################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

def obtencion_de_nombre_png(path):
    img_list = os.listdir(path)

    for file in img_list:
        if not(file.endswith(".png")):
            img_list.remove(file)
    return img_list

def get_metadata(path, organo, organo_len):
    #Obtenemos todas las imágenes en el directorio que sean .png
    #En caso de haber alguna otro tipo de archivo no lo tomará en cuenta para
    #la metadata
    img_list = obtencion_de_nombre_png(path)

    #Creamos una tabla con zeros con el tamaño de la cantidad de imagenes como filas 
    #y una columna donde irán los labels
    zeros = np.zeros( (len(img_list), 1) )
    labels_column = pd.DataFrame(zeros,columns=["labels"])
    
    # Separamos el nombre de la imagen por el separador "-" para obtener sus etiquetas (labels)
    for i, img_name in enumerate(img_list):
        img_name = " ".join(img_name[:organo_len[organo]].split('-'))
        #El resultados nos da algo como esto : 'DT_G0 DC_G0 L_G0 M_G0 N_G0'
        
        labels_column.iloc[i,0] = img_name # localiza las columnas de grado de afectación correspondiente, marca con 1
  
    #Creamos el dataframe metadata para utilizarlos en el modelo de Multi-labe
    metadata = pd.DataFrame()
    metadata = pd.DataFrame(img_list, columns =['img_name'])       
    metadata = metadata.join(labels_column)

    return metadata


def get_img_count(organos, organo_len, caracteristicas, grados, path):
    import warnings
    warnings.filterwarnings("ignore")
    #Definición de las columnas que vamos a usar para cada grado
    columns = ["Organo", "Caracteristica", "Grado"]
    
    #Creamos un dataframe final donde se van a ir uniendo el conteo de cada organo de cada grado de cada caracteristica
    count_final = pd.DataFrame()

    #empezamos la iteración por organos
    for organo in tqdm(organos):

        #En la vaiable count guardaremos la "Cantidad" de imágenes que tiene cada organa para cada caraterística y cada grado
        #Al final la transformaremos a un pd.Series para unirlo al DataFrame final "count_organo" que se unirá al dataframe final
        #"count_final"
        count = {}

        #La variable "count_organo" es donde vamos a guardar el dataframe de cada organo en el siguiente formato
        '''
        Organo              Característica      Grado      Cantidad
        Hepatopancreas            DT              G0          123
        Hepatopancreas            DT              G1          123
        Hepatopancreas            DT              G2          123
        Hepatopancreas            DT              G3          123
        Hepatopancreas            DT              G4          123
        Hepatopancreas            DC              G0          231
        Hepatopancreas            DC              G1          231
        Hepatopancreas            DC              G2          231
        Hepatopancreas            DC              G3          231
        Hepatopancreas            DC              G4          231'''
        count_organo = pd.DataFrame(np.zeros((len(caracteristicas[organo])*len(grados), len(columns))), columns = columns)

        #Lista de imágenes dentro de la carpeta de cada organo
        img_list = obtencion_de_nombre_png(path["img"][organo])
        
        i = 0
        for caracteristica in caracteristicas[organo]:
            for grado in grados:
                count_organo["Organo"][i] = organo
                count_organo["Caracteristica"][i] = caracteristica
                count_organo["Grado"][i] = grado
                i+=1
                
                count[caracteristica, grado] = 0
                for img_name in img_list:
                    img_name = img_name[:organo_len[organo]].split('-')
                    string_to_search = caracteristica + "_" + grado
                    if string_to_search in img_name:
                        count[caracteristica, grado] += 1
                        
        count = pd.DataFrame(pd.Series(count.values()), columns =['Cantidad'])            
        count_organo = count_organo.join(count)
        count_final = pd.concat([count_final, count_organo], ignore_index=True)
    return count_final


######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################## DEFAULT POR fastai###########################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

# Numpy and pandas by default assume a narrow screen - this fixes that
from fastai.vision.all import *
from nbdev.showdoc import *
from ipywidgets import widgets
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

import matplotlib as mpl
# mpl.rcParams['figure.dpi']= 200
mpl.rcParams['savefig.dpi']= 200
mpl.rcParams['font.size']=12

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pd.set_option('display.max_columns',999)
np.set_printoptions(linewidth=200)
torch.set_printoptions(linewidth=200)

import graphviz
def gv(s): return graphviz.Source('digraph G{ rankdir="LR"' + s + '; }')

def get_image_files_sorted(path, recurse=True, folders=None): return get_image_files(path, recurse, folders).sorted()


# +
# pip install azure-cognitiveservices-search-imagesearch

from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth

def search_images_bing(key, term, min_sz=128, max_images=150):    
     params = {'q':term, 'count':max_images, 'min_height':min_sz, 'min_width':min_sz}
     headers = {"Ocp-Apim-Subscription-Key":key}
     search_url = "https://api.bing.microsoft.com/v7.0/images/search"
     response = requests.get(search_url, headers=headers, params=params)
     response.raise_for_status()
     search_results = response.json()    
     return L(search_results['value'])


# -

def search_images_ddg(key,max_n=200):
     """Search for 'key' with DuckDuckGo and return a unique urls of 'max_n' images
        (Adopted from https://github.com/deepanprabhu/duckduckgo-images-api)
     """
     url        = 'https://duckduckgo.com/'
     params     = {'q':key}
     res        = requests.post(url,data=params)
     searchObj  = re.search(r'vqd=([\d-]+)\&',res.text)
     if not searchObj: print('Token Parsing Failed !'); return
     requestUrl = url + 'i.js'
     headers    = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0'}
     params     = (('l','us-en'),('o','json'),('q',key),('vqd',searchObj.group(1)),('f',',,,'),('p','1'),('v7exp','a'))
     urls       = []
     while True:
         try:
             res  = requests.get(requestUrl,headers=headers,params=params)
             data = json.loads(res.text)
             for obj in data['results']:
                 urls.append(obj['image'])
                 max_n = max_n - 1
                 if max_n < 1: return L(set(urls))     # dedupe
             if 'next' not in data: return L(set(urls))
             requestUrl = url + data['next']
         except:
             pass


def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)

# +
from sklearn.tree import export_graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=0, **kwargs):
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, rounded=True,
                      special_characters=True, rotate=False, precision=precision, **kwargs)
    return graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s))


# +
from scipy.cluster import hierarchy as hc

def cluster_columns(df, figsize=(10,6), font_size=12):
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=figsize)
    hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=font_size)
    plt.show()
    
    
def plot_function(f, tx=None, ty=None, title=None, min=-2, max=2, figsize=(6,4)):
    x = torch.linspace(min,max)
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(x,f(x))
    if tx is not None: ax.set_xlabel(tx)
    if ty is not None: ax.set_ylabel(ty)
    if title is not None: ax.set_title(title)
    
    
