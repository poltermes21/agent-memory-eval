"""Neo4j connection helper for Arm D.

Separate from src/db.py: the graph lives in its own container, and is a
projection of `graph_facts` that src/build_graph.py rebuilds for $0.
"""
import logging

from neo4j import GraphDatabase

from src.config import NEO4J_ARMD_PASSWORD, NEO4J_ARMD_URI, NEO4J_ARMD_USER


# Neo4j does not store null properties, so reading r.valid_to warns on every
# call. The read is correct (missing == still valid); only the log line is noise.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def get_driver():
    return GraphDatabase.driver(NEO4J_ARMD_URI, auth=(NEO4J_ARMD_USER, NEO4J_ARMD_PASSWORD))
