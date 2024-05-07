import streamlit as st
import pandas as pd
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter
from kneed import KneeLocator
from scipy.spatial import ConvexHull

COLOR_MAP = {"other": "rgb(190, 190, 190)", 
            "CD15+Tumor": "rgb(73, 176, 248)",
            "CD15-Tumor": "rgb(138, 79, 45)",
            "Tcell": "rgb(235, 74, 148)",
            "Bcell": "rgb(204, 49, 31)",
            "BnTcell": "rgb(236, 95, 42)",
            "Neutrophil": "rgb(0, 40, 245)",
            "Macrophage": "rgb(97, 209, 62)",
            "DC": "rgb(49, 113, 30)"}

def create_new_key(index_list, input_list):
    """Stwórz nowy klucz w słowniku phenotype:celltype"""
    new_key = ''.join(input_list[i] for i in index_list)
    return new_key

def find_graph(bt_cells, patient_data):
    """Funkcja wykonujaca grafy sasiedztwa i slownik mapujacy indeksy ze spojnych skladowych na patient_data"""

    coor_dict = dict()
    j = 0
    for i, row in bt_cells.iterrows():
        coor_dict[j] = i
        j+=1

    bt_cells_nc = bt_cells[["nucleus.x","nucleus.y"]].values
    all_cells_nc = patient_data[["nucleus.x","nucleus.y"]].values

    graph_bt = radius_neighbors_graph(bt_cells_nc, 30, mode='distance', include_self='auto')
    graph_all = radius_neighbors_graph(all_cells_nc, 30, mode='distance', include_self='auto')

    return graph_bt, graph_all, coor_dict

def cc(graph_bt, graph_all, coor_dict, patient_data):
    """Funkcja znajdujaca spojne skladowe zwracajaca liste indeksow i komorek"""

    n_components, labels = connected_components(csgraph=graph_bt, directed=False)

    components = [[] for _ in range(n_components)]

    for node, label in enumerate(labels):
        components[label].append(node)

    long_components = [component for component in components if len(component) > 20] # wybieram tylko te odpowiednio dlugie 

    long_components_reindexed = [[coor_dict[node] for node in component] for component in long_components] # mapuje indeksy na te z patient_data

    long_components_reindexed_cells = []

    for i in range(len(long_components_reindexed)):

        component = long_components_reindexed[i]
        for j in range(len(component)):
            node = component[j]
            neighbors = graph_all[node].nonzero()[1] 
            component.extend(neighbors) # dodaje sasiadow z pelnego grafu

        component = list(set(component)) # usuń duplikaty
        long_components_reindexed[i] = np.copy(component)  

        for k in range(len(component)):
            node = component[k] 
            component[k] = patient_data.iloc[node]['celltype'] # celltypes
        
        long_components_reindexed_cells.append(component)
       
    return long_components_reindexed, long_components_reindexed_cells

def create_dictionary(slownik_file):
    """Funkcja tworzaca słownik phenotype:celltype"""
    slownik = pd.read_csv(slownik_file)
    phenotype = []
    for key in slownik["phenotype"]:
        new_key = ""
        for letter in key:
            if letter == "+" or letter == "-":
                new_key += letter + "_" 
            else:
                new_key += letter
        splitted_key = new_key.split("_")[:-1]
        index_list =  [1, 5, 4, 0, 3, 2]
        new_key = create_new_key(index_list, splitted_key)
        phenotype.append(new_key)
    slownik["phenotype"] = phenotype
    return slownik

def process_data(slownik, patient_file):
    """Funkcja tworzaca DataFrame z danych pacjenta"""
    patient = pd.read_csv(patient_file)
    cell_dict = slownik.set_index('phenotype')['celltype'].to_dict()
    patient["celltype"] = patient["phenotype"].apply(lambda x: cell_dict.get(x))
    return patient

def prepare_vectors(comps, cell_types, file_id):
    """Funkcja tworzaca wektory udzialu procentowego komorek"""
    vectors = []

    for comp in comps:
        comp_counter = Counter(comp)
        vector = np.zeros(len(cell_types))
        total_cells = sum(comp_counter.values())
        for idx, cell_type in enumerate(cell_types):
            if total_cells != 0:
                percentage = comp_counter[cell_type] / total_cells * 100
            else:
                percentage = 0
            vector[idx] = percentage
        vectors.append((file_id, vector))
    
    return vectors

def vectors_graph(all_vectors, cell_types, clustered, cluster_indices, include_clusters=True):
    """Funkcja tworzaca wykresy slupkowe przed i po sklastrowaniu TLS-ów"""
    def calculate_percentage(vector):
        total = sum(vector)
        percentages = [round((count / total) * 100, 2) for count in vector]
        return percentages
    
    percentage_vectors = [(file_name, calculate_percentage(vector)) for file_name, vector in all_vectors]

    df = pd.DataFrame([vec for _, vec in percentage_vectors], columns=cell_types)
    df['Numer wektora'] = df.index
    df['file_name'] = [name for name, _ in percentage_vectors]
    
    if include_clusters:
        df['Cluster'] = cluster_indices  
        id_vars = ['Numer wektora', 'file_name', 'Cluster']
        hover_data = ['Typ komórki', 'Procentowy udział', 'file_name', 'Cluster']
    else:
        id_vars = ['Numer wektora', 'file_name']
        hover_data = ['Typ komórki', 'Procentowy udział', 'file_name']

    fig = px.bar(df.melt(id_vars=id_vars, var_name='Typ komórki', value_name='Procentowy udział'),
                x='Numer wektora', y='Procentowy udział', color='Typ komórki',
                title=f'Procentowy udział komórek per TLS {clustered}',
                labels={'Numer wektora': 'TLS', 'Procentowy udział': 'Procentowy udział [%]'},
                hover_data= hover_data)

    if include_clusters:
        cluster_changes = [i for i in range(1, len(cluster_indices)) if cluster_indices[i] != cluster_indices[i - 1]]
        for change_point in cluster_changes:
            fig.add_shape(type="line", x0=change_point - 0.5, y0=0, x1=change_point - 0.5, y1=100, line=dict(color="White", width=4))

    st.plotly_chart(fig)

def biopsy_graph(connected_comps, patient_data, file_id):
    """Funkcja rysujaca dwuwymiarowy wykres biopsji, takze z ogniskami"""
    fig = px.scatter(patient_data, x='nucleus.x', y='nucleus.y', color='celltype',
                    color_discrete_map=COLOR_MAP,
                    opacity=0.3, size_max=5)
    
    fig.update_layout(title=f"Dwuwymiarowy wykres biopsji dla próbki {file_id}")

    st.plotly_chart(fig)


    fig = px.scatter(patient_data, x='nucleus.x', y='nucleus.y', color='celltype',
                    color_discrete_map=COLOR_MAP,
                    opacity=0.013, size_max=5)

    fig.update_layout(title=f"Dwuwymiarowy wykres biopsji z zaznaczonymi ogniksami rakowymi dla próbki {file_id}")

    for comp_idx, comp in enumerate(connected_comps):
        comp_data = patient_data[patient_data.index.isin(comp)]
        trace = px.scatter(comp_data, x='nucleus.x', y='nucleus.y', color_discrete_sequence=['red'], 
                        opacity=1.0, size_max=10)
        fig.add_trace(trace.data[0])

    st.plotly_chart(fig)

def show_ccs(patient_data, connected_comps, patient_file):
    """Funkcja wizualizujaca TLS-y"""
    for comp_idx, comp in enumerate(connected_comps):
        comp_data = patient_data.iloc[comp]
        
        fig = px.scatter(comp_data, x='nucleus.x', y='nucleus.y', color='celltype',
                        color_discrete_map=COLOR_MAP,
                        opacity=0.7, size_max=10)

        fig.update_layout(title=f"Wykres dla {comp_idx + 1}. TLS pacjenta {patient_file.name[:4]}")
        st.plotly_chart(fig)

def vis_clusters(ccs, patient_dict):
    """Funkcja wizualizujaca klastry nakladajac je na siebie"""

    for i, cluster in enumerate(ccs):
        centroids_info = pd.DataFrame(columns=['ID pliku', 'Centrum X', 'Centrum Y'])
        cluster_data = pd.DataFrame(columns=['nucleus.x', 'nucleus.y', 'celltype'])

        for file_id, indices in cluster:
            file_data = patient_dict[file_id].iloc[indices, :]

            centroid_x = file_data['nucleus.x'].mean()
            centroid_y = file_data['nucleus.y'].mean()

            centroids_info = centroids_info._append({'ID pliku': file_id, 'Centrum X': centroid_x, 'Centrum Y': centroid_y}, ignore_index=True)

            file_data['nucleus.x'] -= centroid_x
            file_data['nucleus.y'] -= centroid_y

            max_distance = max(abs(file_data['nucleus.x']).max(), abs(file_data['nucleus.y']).max())
            file_data['nucleus.x'] /= max_distance
            file_data['nucleus.y'] /= max_distance

            cluster_data = pd.concat([cluster_data, file_data])

        fig = px.scatter(cluster_data,
                         x="nucleus.x",
                         y="nucleus.y",
                         color='celltype',
                         color_discrete_map=COLOR_MAP)

        fig.update_layout(
            title=f'TLSy dla klastra {i}',
            xaxis_title='Nucleus X',
            yaxis_title='Nucleus Y',
            showlegend=True,
            autosize=False,
            width=800,
            height=600
        )

        st.plotly_chart(fig)

        st.table(centroids_info)

def cluster_graphs(all_vectors, all_cell_types, all_ccs):
    """Funkcja klastrujaca i wizualizujaca klastry"""

    features = np.array([vector[1] for vector in all_vectors])
    distortions = []
    max_clusters = len(features)

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        distortions.append(kmeans.inertia_)  

    kl = KneeLocator(range(1, max_clusters + 1), distortions, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(features)

    labels = kmeans.labels_

    indeksy_posortowane = np.argsort(labels)

    sorted_vectors = [all_vectors[i] for i in indeksy_posortowane]


    vectors_graph(sorted_vectors, all_cell_types, "po klasteryzacji", sorted(labels), True)
    
    features = np.array([vector[1] for vector in all_vectors])

    pca = PCA(n_components=2)  
    components = pca.fit_transform(features)

    fig = go.Figure()

    for i, cluster in enumerate(np.unique(labels)):
        cluster_indices = np.where(labels == cluster)[0]
        fig.add_trace(go.Scatter(x=components[cluster_indices, 0], y=components[cluster_indices, 1], mode='markers', name=f'Klaster {cluster}'))

        if len(cluster_indices) > 2:  
            hull = ConvexHull(components[cluster_indices], qhull_options='QJ Pp')
            hull_points = components[cluster_indices][hull.vertices]  
            hull_points = np.append(hull_points, [hull_points[0]], axis=0)  

            fig.add_trace(go.Scatter(x=hull_points[:, 0], y=hull_points[:, 1], fill='toself', mode='lines', line_color=fig.data[i]['marker']['color'], fillcolor=fig.data[i]['marker']['color'], name=f'Area Cluster {cluster}', showlegend=False))

    fig.update_layout(title='Wizualizacja PCA klastrów', xaxis_title='PCA 1', yaxis_title='PCA 2')

    
    st.plotly_chart(fig)

    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    cluster_indices = {}
    for cluster_label in unique_clusters:
        cluster_indices[cluster_label] = np.where(labels == cluster_label)[0]

    clusters = []
    for cluster_label, indices in cluster_indices.items():
        cluster_files = [(all_vectors[index][0], all_ccs[index]) for index in indices]
        clusters.append(cluster_files)

    return clusters

def main():
    st.title("Analiza danych IF")

    all_cell_types = ["other", "CD15+Tumor", "CD15-Tumor", "Tcell", "Bcell", "BnTcell", "Neutrophil", "Macrophage", "DC"]

    uploaded_files = st.file_uploader("Wybierz pliki CSV pacjenta", type="csv", accept_multiple_files=True)
    
    perform_biopsy = st.checkbox("Wykonaj wykresy biopsji")

    button_vectors = st.checkbox("Wykres słupkowy niesklastrowanych TLS")

    button_ccs = st.checkbox("Wykresy wszystkich TLSów na osobnych wykresach")

    button_clusters = st.checkbox("Klastruj")

    button_vis_clusters = st.checkbox("Wykres przestrzennego podobieństwa klastrów")

    button_start = st.checkbox("Rozpocznij analizę")

    
    if uploaded_files and button_start:
        slownik_file = "IF1_phen_to_cell_mapping.csv"
        slownik = create_dictionary(slownik_file)

        all_vectors = []
        all_ccs =[]
        patient_dict = dict()

        for patient_file in uploaded_files:

            st.write(f"Analiza pliku: {patient_file.name}")

            patient_data = process_data(slownik, patient_file)

            patient_dict[patient_file.name[:4]] = patient_data

            bt_cells = patient_data[patient_data['celltype'].isin(['Bcell', 'BnTcell', 'Tcell'])]

            graph_bt, graph_all, coor_dict = find_graph(bt_cells, patient_data)

            connected_comps, connected_comps_cells = cc(graph_bt, graph_all, coor_dict, patient_data)

            all_ccs.extend(connected_comps)

            vectors_matrix = prepare_vectors(connected_comps_cells, all_cell_types, patient_file.name[:4])

            if perform_biopsy:
                biopsy_graph(connected_comps, patient_data, patient_file.name[:4])
            
            if button_ccs:
                show_ccs(patient_data, connected_comps, patient_file)

            all_vectors.extend(vectors_matrix)

        if button_vectors:
            vectors_graph(all_vectors, all_cell_types, "przed klasteryzacją", [], False)
        if button_clusters:
            clusters= cluster_graphs(all_vectors, all_cell_types, all_ccs)
            if button_vis_clusters:
                vis_clusters(clusters, patient_dict)
        


if __name__ == "__main__":
    main()
