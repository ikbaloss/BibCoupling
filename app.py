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
    required = ['Authors', 'Year', 'Title', 'Abstract', 'References']
    for col in required:
        if col not in df.columns:
            df[col] = ""
            
    df['Year'] = df['Year'].fillna(0).astype(int).astype(str)
    df['Title'] = df['Title'].fillna("Unknown Title")
    df['Authors'] = df['Authors'].fillna("Unknown")
    
    def make_label(row):
        first_author = row['Authors'].split(',')[0].strip()
        return f"{first_author} ({row['Year']})"
    
    df['Node_Label'] = df.apply(make_label, axis=1)
    
    # Keep the raw case-preserved list for pristine rendering in popups, and a lower-case set for matching
    df['Ref_Set'] = df['References'].fillna("").apply(
        lambda x: set([r.strip() for r in x.split(';') if r.strip()])
    )
    return df

def build_bibliographic_coupling_network(df):
    """Constructs the base full network using shared references (Bibliographic Coupling)."""
    G = nx.Graph()
    
    for _, row in df.iterrows():
        G.add_node(
            row['Node_Label'], 
            authors=row['Authors'], 
            year=row['Year'], 
            title=row['Title'], 
            abstract=row['Abstract'],
            refs=row['Ref_Set'] # Store references inside the node attributes
        )
        
    nodes = list(df['Node_Label'])
    ref_sets = list(df['Ref_Set'])
    num_nodes = len(nodes)
    
    for i in range(num_nodes):
        # Create lowercase lookup sets for accurate intersection matching
        set_i_lower = {r.lower(): r for r in ref_sets[i]}
        for j in range(i + 1, num_nodes):
            set_j_lower = {r.lower() for r in ref_sets[j]}
            
            # Find overlapping keys
            shared_keys = set(set_i_lower.keys()).intersection(set_j_lower)
            weight = len(shared_keys)
            
            if weight > 0:
                # Extract the original case-preserved reference strings
                actual_shared_refs = [set_i_lower[k] for k in shared_keys]
                G.add_edge(nodes[i], nodes[j], weight=weight, shared_refs=actual_shared_refs)
                
    return G

def filter_network(G, min_weight):
    """Filters edges below threshold and drops any resulting isolated nodes."""
    F = G.copy()
    broken_edges = [(u, v) for u, v, d in F.edges(data=True) if d.get('weight', 1) < min_weight]
    F.remove_edges_from(broken_edges)
    
    # Modernized NetworkX 3.x+ isolate dropping protocol
    isolated_nodes = list(nx.isolates(F))
    F.remove_nodes_from(isolated_nodes)
    
    return F

def apply_leiden_clustering(G):
    """Applies Leiden clustering via igraph backend and maps partition results to NetworkX."""
    if len(G.nodes) == 0:
        return G
        
    mapping = {node: i for i, node in enumerate(G.nodes())}
    inv_mapping = {i: node for node, i in mapping.items()}
    
    ig_g = ig.Graph(len(G.nodes), directed=False)
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_g.add_edges(edges)
    
    weights = [d.get('weight', 1.0) for u, v, d in G.edges(data=True)]
    if weights:
        ig_g.es['weight'] = weights
        
    partition = la.find_partition(ig_g, la.ModularityVertexPartition, weights='weight', seed=42)
    
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

if 'base_graph' not in st.session_state:
    st.session_state.base_graph = None
if 'clustered_graph' not in st.session_state:
    st.session_state.clustered_graph = None
if 'is_clustered' not in st.session_state:
    st.session_state.is_clustered = False
if 'prev_weight' not in st.session_state:
    st.session_state.prev_weight = 1

st.sidebar.header("📁 Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Scopus Export (CSV Format)", type=["csv"])

if uploaded_file is not None:
    if st.session_state.base_graph is None:
        with st.spinner("Processing Scopus parsing & reference tracking..."):
            raw_df = pd.read_csv(uploaded_file)
            parsed_df = parse_scopus_data(raw_df)
            st.session_state.base_graph = build_bibliographic_coupling_network(parsed_df)
            st.session_state.clustered_graph = st.session_state.base_graph.copy()
            st.session_state.is_clustered = False

    st.sidebar.header("⚙️ Network Topology Parameters")
    min_edge_weight = st.sidebar.number_input(
        "Minimum Edge Weight Threshold", 
        min_value=1, 
        value=1, 
        step=1
    )
    
    # NEW UI CONTROLS FOR VISIBILITY
    st.sidebar.header("👁️ Display Options")
    show_edge_labels = st.sidebar.checkbox("Show Edge Weights on Map", value=False)
    
    if min_edge_weight != st.session_state.prev_weight:
        st.session_state.is_clustered = False
        st.session_state.prev_weight = min_edge_weight

    working_graph = filter_network(st.session_state.base_graph, min_edge_weight)
    
    st.sidebar.header("🤖 Community Segmentation")
    cluster_btn = st.sidebar.button(
        "Run Leiden Clustering", 
        disabled=st.session_state.is_clustered or len(working_graph.nodes) == 0
    )
    
    if cluster_btn:
        with st.spinner("Executing optimized community partitions..."):
            working_graph = apply_leiden_clustering(working_graph)
            st.session_state.clustered_graph = working_graph
            st.session_state.is_clustered = True
            st.rerun()

    cluster_options = ["All"]
    if st.session_state.is_clustered:
        working_graph = st.session_state.clustered_graph
        unique_clusters = sorted(list(set([d.get('cluster', 'Unclustered') for n, d in working_graph.nodes(data=True)])))
        cluster_options.extend(unique_clusters)
        
    selected_cluster = st.sidebar.selectbox("Filter Target Workspace by Cluster", options=cluster_options, disabled=not st.session_state.is_clustered)

    if st.session_state.is_clustered and selected_cluster != "All":
        target_nodes = [n for n, d in working_graph.nodes(data=True) if d.get('cluster') == selected_cluster]
        display_graph = working_graph.subgraph(target_nodes).copy()
    else:
        display_graph = working_graph.copy()

    # Robust Multi-Stage Fail-Safe Centrality Computation
    eigen_centrality = {}
    if len(display_graph.nodes) > 0:
        try:
            eigen_centrality = nx.eigenvector_centrality(display_graph, max_iter=1000, weight=None)
        except nx.PowerIterationFailedConvergence:
            try:
                eigen_centrality = nx.eigenvector_centrality_numpy(display_graph, weight=None)
            except:
                eigen_centrality = nx.degree_centrality(display_graph)
        except:
            eigen_centrality = nx.degree_centrality(display_graph)
                
        nx.set_node_attributes(display_graph, eigen_centrality, 'centrality')

    # -----------------------------------------------------------------------------
    # 3. GRAPH VISUALIZATION & METRICS DISPLAY LAYOUT
    # -----------------------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Network Visualization Map — Displaying {len(display_graph.nodes)} Active Nodes")
        st.caption("💡 Pro-Tip: Hover over or click an edge to inspect shared reference metadata stacks.")
        
        if len(display_graph.nodes) > 0:
            pv_net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
            
            cluster_color_map = {
                "Cluster 1": "#1f77b4", "Cluster 2": "#ff7f0e", "Cluster 3": "#2ca02c",
                "Cluster 4": "#d62728", "Cluster 5": "#9467bd", "Cluster 6": "#8c564b"
            }
            
            for node, data in display_graph.nodes(data=True):
                c_val = data.get('cluster', 'Unclustered')
                color = cluster_color_map.get(c_val, "#7f7f7f") if st.session_state.is_clustered else "#1f77b4"
                size_factor = 10 + (data.get('centrality', 0.1) * 40)
                
                hover_title = f"<b>{node}</b><br>Title: {data.get('title')}<br>Cluster: {c_val}"
                pv_net.add_node(node, label=node, title=hover_title, color=color, size=size_factor)
                
            for u, v, d in display_graph.edges(data=True):
                w = d.get('weight', 1)
                shared_list = d.get('shared_refs', [])
                
                # Format an elegant HTML breakdown for the hover tooltip popup box
                refs_html = "<br>".join([f"• {r}" for r in sorted(shared_list)])
                edge_popup_title = f"<b>Shared References ({w}):</b><br><div style='max-height:200px; overflow-y:auto; font-size:11px;'>{refs_html}</div>"
                
                # Determine text label display status based on checkbox state
                edge_label_text = str(w) if show_edge_labels else ""
                
                pv_net.add_edge(
                    u, v, 
                    value=w, 
                    label=edge_label_text, 
                    title=edge_popup_title,
                    color="#cccccc"
                )
                
            pv_net.toggle_physics(True)
            pv_net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
            
            html_data = pv_net.generate_html()
            components.html(html_data, height=620, scrolling=False)
        else:
            st.warning("The operational configuration criteria contains 0 valid graphical nodes.")

    with col2:
        st.subheader("Data Export")
        if len(working_graph.nodes) > 0:
            gml_buffer = io.BytesIO()
            exportable_graph = working_graph.copy()
            
            # 1. Cleanse complex data types out of node profiles
            for node, data in exportable_graph.nodes(data=True):
                if 'refs' in data:
                    del data['refs']
            
            # 2. Cleanse complex data types out of edge profiles
            for u, v, d in exportable_graph.edges(data=True):
                if 'shared_refs' in d:
                    del d['shared_refs']
                    
            # 3. Now it is 100% safe to run the NetworkX GML compiler
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
