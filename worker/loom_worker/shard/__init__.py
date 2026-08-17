"""Pipeline-stage executor: runs a contiguous slice of a model's layers.

This is what makes Loom's core idea work — a model too large for one GPU is
split across several nodes, each running `[start_layer, end_layer)` and passing
hidden states to the next stage.

Design follows Parallax's data-plane roles (see loader.py / executor.py for
attribution): the first stage owns embeddings and the client request, middle
stages transform hidden states, the last stage owns the LM head and sampling
and sends the sampled token straight back to the head.
"""
