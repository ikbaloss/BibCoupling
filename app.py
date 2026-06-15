import streamlit as st
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
from pyvis.network import Network
import streamlit.components.v1 as components
import io

# -----------------------------------------------------------------------------
# 1. CORE HELPER FUNCTIONS (Graph Logic & Processing)
# -----------------------------------------------------------------------------

def parse_scopus_data(df):
    """Parses Scopus data to extract references and construct unique document identifiers."""
    # Ensure necessary columns exist
    required = ['Authors', 'Year', 'Title', 'Abstract', 'References']
    for col in required:
        if col not in df.columns:
            df[col] = ""
            
    df['Year'] = df['Year'].fillna(0).astype(int).astype(str)
    df['Title'] = df['Title'].fillna("Unknown Title")
    df['Authors'] = df['Authors'].fillna("Unknown")
    
    # Generate a recognizable label: "Author (Year)"
    def make_label(row):
        first_author = row['Authors'].split(',')[0].strip()
        return f"{first_author} ({row['Year']})"
    
    df['Node_Label'] = df.apply(make_label, axis=1)
    
    # Process references into sets for fast intersection mapping
    df['Ref_Set'] = df['References'].fillna("").apply(
        lambda x: set([r.strip().lower() for r in x.split(';') if r.strip()])
    )
    return df

def build_bibliographic_coupling_network(df):
    """Constructs the base full network using shared references (Bibliographic Coupling)."""
    G = nx.Graph()
    
    # Add nodes with metadata
    for _, row in df.iterrows():
        G.add_node(
            row['Node_Label'], 
            authors=row['Authors'], 
            year=row['Year'], 
            title=row['Title'], 
            abstract=row['Abstract']
        )
        
    # Add edges based on shared reference count (Bibliographic Coupling Weight)
    nodes = list(df['Node_Label'])
    ref_sets = list(df['Ref_Set'])
    num_nodes = len(nodes)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            shared_refs = ref_sets[i].intersection(ref_sets[j])
            weight = len(shared_refs)
            if weight > 0:
                G.add_edge(nodes[i], nodes[j], weight=weight)
                
    return G

def filter_network(G, min_weight):
    """Filters edges below threshold and drops any resulting isolated nodes."""
    # Copy graph to prevent mutation issues
    F = G.copy()
    
    # Remove weak edges
    broken_edges = [(u, v) for u, v, d in F.edges(data=True) if d.get('weight', 1) < min_weight]
    F.remove_edges_from(broken_edges)
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(F))
    F.remove_nodes_from(isolated_nodes)
    
    return F

def apply_leiden_clustering(G):
    """Applies Leiden clustering via igraph backend and maps partition results to NetworkX."""
    if len(G.nodes) == 0:
        return G
        
    # Convert NetworkX graph to igraph format for Leiden processing
    mapping = {node: i for i, node in enumerate(G.nodes())}
    inv_mapping = {i: node for node, i in mapping.items()}
    
    ig_g = ig.Graph(len(G.nodes), directed=False)
    
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_g.add_edges(edges)
    
    weights = [d.get('weight', 1.0) for u, v, d in G.edges(data=True)]
    if weights:
        ig_g.es['weight'] = weights
        
    # Compute Leiden partition using Modularities Optimization
    partition = la.find_partition(ig_g, la.ModularityVertexPartition, weights='weight', seed=42)
    
    # Map back clusters to NetworkX node attributes
    for cluster_idx, community in enumerate(partition):
        for node_idx in community:
            node_name = inv_mapping[node_idx]
            G.nodes[node_name]['cluster'] = f"Cluster {cluster_idx + 1}"
            
    return G

# -----------------------------------------------------------------------------
# 2. STREAMLIT USER INTERFACE & STATE MANAGEMENT
# -----------------------------------------------------------------------------

st.set_page_config(page_title="VOSviewer Bibliographic Coupling Replica", layout="wide")
st.title("🔬 Bibliographic Coupling Network Analyzer")
st.caption("A web tool replicating VOSviewer coupling workflows and Gephi centrality diagnostics.")

# Session State Initialization
if 'base_graph' not in st.session_state:
    st.session_state.base_graph = None
if 'clustered_graph' not in st.session_state:
    st.session_state.clustered_graph = None
if 'is_clustered' not in st.session_state:
    st.session_state.is_clustered = False
if 'prev_weight' not in st.session_state:
    st.session_state.prev_weight = 1

# Sidebar Panel for Configs
st.sidebar.header("📁 Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Scopus Export (CSV Format)", type=["csv"])

if uploaded_file is not None:
    # Load and build model once
    if st.session_state.base_graph is None:
        with st.spinner("Processing Scopus parsing & reference tracking..."):
            raw_df = pd.read_csv(uploaded_file)
            parsed_df = parse_scopus_data(raw_df)
            st.session_state.base_graph = build_bibliographic_coupling_network(parsed_df)
            st.session_state.clustered_graph = st.session_state.base_graph.copy()
            st.session_state.is_clustered = False

    # Configuration Parameters
    st.sidebar.header("⚙️ Network Topology Parameters")
    min_edge_weight = st.sidebar.number_input(
        "Minimum Edge Weight Threshold", 
        min_value=1, 
        value=1, 
        step=1
    )
    
    # Handle Threshold Changes (Reset Clustering State)
    if min_edge_weight != st.session_state.prev_weight:
        st.session_state.is_clustered = False
        st.session_state.prev_weight = min_edge_weight

    # Extract working subgraph based on edge constraints
    working_graph = filter_network(st.session_state.base_graph, min_edge_weight)
    
    # Clustering Actions
    st.sidebar.header("🤖 Community Segmentation")
    cluster_btn = st.sidebar.button(
        "Run Leiden Clustering", 
        disabled=st.session_state.is_clustered or len(working_graph.nodes) == 0
    )
    
    if cluster_btn:
        with st.spinner("Executing optimized community partitions..."):
            working_graph = apply_leiden_clustering(working_graph)
            # Store partition metadata back into persistent state tracking
            st.session_state.clustered_graph = working_graph
            st.session_state.is_clustered = True
            st.rerun()

    # Dynamic Cluster Focus Filter Selection
    cluster_options = ["All"]
    if st.session_state.is_clustered:
        working_graph = st.session_state.clustered_graph
        unique_clusters = sorted(list(set([d.get('cluster', 'Unclustered') for n, d in working_graph.nodes(data=True)])))
        cluster_options.extend(unique_clusters)
        
    selected_cluster = st.sidebar.selectbox("Filter Target Workspace by Cluster", options=cluster_options, disabled=not st.session_state.is_clustered)

    # Filter Down Workspace Graph based on Cluster Filter selection
    if st.session_state.is_clustered and selected_cluster != "All":
        target_nodes = [n for n, d in working_graph.nodes(data=True) if d.get('cluster') == selected_cluster]
        display_graph = working_graph.subgraph(target_nodes).copy()
    else:
        display_graph = working_graph.copy()

    # Network Metrics Calculations
    eigen_centrality = {}
    if len(display_graph.nodes) > 0:
        if st.session_state.is_clustered and selected_cluster != "All":
            # Compute sub-cluster specific eigenvector centrality matching Gephi behaviors
            try:
                eigen_centrality = nx.eigenvector_centrality_numpy(display_graph, weight='weight')
            except:
                eigen_centrality = {node: 0.0 for node in display_graph.nodes}
        else:
            # Fallback uniform centrality metric for general viewports
            try:
                eigen_centrality = nx.eigenvector_centrality_numpy(display_graph, weight='weight')
            except:
                eigen_centrality = {node: 0.0 for node in display_graph.nodes}
                
        nx.set_node_attributes(display_graph, eigen_centrality, 'centrality')

    # -----------------------------------------------------------------------------
    # 3. GRAPH VISUALIZATION & METRICS DISPLAY LAYOUT
    # -----------------------------------------------------------------------------
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Network Visualization Map — Displaying {len(display_graph.nodes)} Active Nodes")
        
        if len(display_graph.nodes) > 0:
            # Generate Interactive Pyvis Ecosystem
            pv_net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
            
            # Setup cluster unique visual colors
            cluster_color_map = {
                "Cluster 1": "#1f77b4", "Cluster 2": "#ff7f0e", "Cluster 3": "#2ca02c",
                "Cluster 4": "#d62728", "Cluster 5": "#9467bd", "Cluster 6": "#8c564b"
            }
            
            for node, data in display_graph.nodes(data=True):
                c_val = data.get('cluster', 'Unclustered')
                color = cluster_color_map.get(c_val, "#7f7f7f") if st.session_state.is_clustered else "#1f77b4"
                
                # Dynamic node scale maps to centrality metrics
                size_factor = 10 + (data.get('centrality', 0.1) * 40)
                
                hover_title = f"<b>{node}</b><br>Title: {data.get('title')}<br>Cluster: {c_val}"
                
                pv_net.add_node(
                    node, 
                    label=node, 
                    title=hover_title, 
                    color=color, 
                    size=size_factor
                )
                
            for u, v, d in display_graph.edges(data=True):
                pv_net.add_edge(u, v, value=d.get('weight', 1))
                
            # Physics performance optimization structures
            pv_net.toggle_physics(True)
            pv_net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
            
            html_data = pv_net.generate_html()
            components.html(html_data, height=620, scrolling=False)
        else:
            st.warning("The operational configuration criteria contains 0 valid graphical nodes.")

    with col2:
        st.subheader("Data Export")
        
        # GML Formatting Export Engine
        if len(working_graph.nodes) > 0:
            gml_buffer = io.BytesIO()
            # Clean non-standard complex metadata structures prior to binary conversion execution
            exportable_graph = working_graph.copy()
            nx.write_gml(exportable_graph, gml_buffer)
            
            st.download_button(
                label="📥 Export Full GML File",
                data=gml_buffer.getvalue(),
                file_name="bibliographic_coupling_network.gml",
                mime="application/gml+xml"
            )
        else:
            st.write("No viable map metrics ready to pipeline to GML format.")

    # -----------------------------------------------------------------------------
    # 4. RANKED TABULAR VIEWPORT ENGINE (Top N Records Section)
    # -----------------------------------------------------------------------------
    
    if st.session_state.is_clustered and selected_cluster != "All" and len(display_graph.nodes) > 0:
        st.write("---")
        st.subheader("📋 Top Centrality Cluster Records")
        
        top_n = st.number_input("Top N Nodes Selection Size", min_value=1, max_value=len(display_graph.nodes), value=5)
        show_table = st.button("Show Top N Data")
        
        if show_table:
            # Sort node records based on standard local cluster Centrality indexes
            sorted_nodes = sorted(eigen_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            table_rows = []
            for node, weight in sorted_nodes:
                nd_attribs = display_graph.nodes[node]
                table_rows.append({
                    "Authors": nd_attribs.get('authors'),
                    "Year": nd_attribs.get('year'),
                    "Title": nd_attribs.get('title'),
                    "Abstract": nd_attribs.get('abstract'),
                    "Eigenvector Centrality": round(weight, 5)
                })
                
            report_df = pd.DataFrame(table_rows)
            st.dataframe(report_df, use_container_width=True)
else:
    st.info("System awaiting ingestion parameters. Please upload a Scopus data CSV format to seed network configurations.")
