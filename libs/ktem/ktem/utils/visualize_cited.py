from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import umap
import logging

from kotaemon.base import BaseComponent
from kotaemon.embeddings import BaseEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VISUALIZATION_SETTINGS = {
    "Original Query": {"color": "red", "opacity": 1, "symbol": "cross", "size": 15},
    "Retrieved": {"color": "green", "opacity": 1, "symbol": "circle", "size": 10},
    "Chunks": {"color": "blue", "opacity": 0.4, "symbol": "circle", "size": 10},
    "Sub-Questions": {"color": "purple", "opacity": 1, "symbol": "star", "size": 15},
}


class CreateCitationVizPipeline(BaseComponent):
    """Creating PlotData for visualizing query results"""

    embedding: BaseEmbeddings
    projector: Optional[umap.UMAP] = None

    def _set_up_umap(self, embeddings: np.ndarray) -> Optional[umap.UMAP]:
        N = len(embeddings)
        if N < 2:
            logger.warning("Not enough data points for UMAP visualization.")
            return None
        # Set n_neighbors to min(15, N-1) to ensure k < N
        n_neighbors = min(15, max(2, N - 1))
        try:
            umap_transform = umap.UMAP(n_neighbors=n_neighbors).fit(embeddings)
            logger.info(f"UMAP fitted with n_neighbors={n_neighbors}")
            return umap_transform
        except Exception as e:
            logger.error(f"UMAP Error: {e}")
            return None

    def _project_embeddings(self, embeddings: np.ndarray, umap_transform: Optional[umap.UMAP]) -> np.ndarray:
        if umap_transform is None:
            logger.warning("UMAP transformer is None. Returning zero embeddings.")
            return np.zeros((len(embeddings), 2))
        try:
            umap_embeddings = umap_transform.transform(embeddings)
            logger.info("UMAP transformation successful.")
            return umap_embeddings
        except Exception as e:
            logger.error(f"UMAP Transform Error: {e}")
            return np.zeros((len(embeddings), 2))

    def _get_projections(self, embeddings: np.ndarray, umap_transform: Optional[umap.UMAP]) -> Tuple[np.ndarray, np.ndarray]:
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for projection.")
            return np.array([]), np.array([])
        projections = self._project_embeddings(embeddings, umap_transform) if umap_transform else np.zeros((len(embeddings), 2))
        x = projections[:, 0]
        y = projections[:, 1]
        return x, y

    def _prepare_projection_df(
        self,
        document_projections: Tuple[np.ndarray, np.ndarray],
        document_text: List[str],
        plot_size: int = 3,
    ) -> pd.DataFrame:
        """Prepares a DataFrame for visualization from projections and texts.

        Args:
            document_projections (Tuple[np.ndarray, np.ndarray]):
                Tuple of X and Y coordinates of document projections.
            document_text (List[str]): List of document texts.
        """
        x, y = document_projections
        if len(x) != len(document_text):
            logger.error("Number of projections does not match number of documents.")
            return pd.DataFrame()
        df = pd.DataFrame({"x": x, "y": y})
        df["document"] = document_text
        df["document_cleaned"] = df.document.str.wrap(50).apply(
            lambda x: x.replace("\n", "<br>")[:512] + "..."
        )
        df["size"] = plot_size
        df["category"] = "Retrieved"
        return df

    def _plot_embeddings(self, df: pd.DataFrame) -> go.Figure:
        """
        Creates a Plotly figure to visualize the embeddings.

        Args:
            df (pd.DataFrame): DataFrame containing the data to visualize.

        Returns:
            go.Figure: A Plotly figure object for visualization.
        """
        fig = go.Figure()

        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            settings = VISUALIZATION_SETTINGS.get(
                category,
                {"color": "grey", "opacity": 1, "symbol": "circle", "size": 10},
            )
            fig.add_trace(
                go.Scatter(
                    x=category_df["x"],
                    y=category_df["y"],
                    mode="markers",
                    name=category,
                    marker=dict(
                        color=settings["color"],
                        opacity=settings["opacity"],
                        symbol=settings["symbol"],
                        size=settings["size"],
                        line_width=0,
                    ),
                    hoverinfo="text",
                    text=category_df["document_cleaned"],
                )
            )

        fig.update_layout(
            height=500,
            legend=dict(y=100, x=0.5, xanchor="center", yanchor="top", orientation="h"),
        )
        return fig

    def run(self, context: List[str], question: str) -> Optional[go.Figure]:
        # Embed contexts
        embed_contexts = self.embedding(context)  # Assuming self.embedding returns list of objects with 'embedding' attribute
        if not embed_contexts:
            logger.error("No embeddings returned for context.")
            return None
        context_embeddings = np.array([d.embedding for d in embed_contexts])

        # Set up UMAP
        self.projector = self._set_up_umap(embeddings=context_embeddings)

        # Project context embeddings
        context_projections = self._get_projections(context_embeddings, self.projector)
        viz_base_df = self._prepare_projection_df(
            document_projections=(context_projections[0], context_projections[1]),
            document_text=context
        )

        # Embed query
        embed_query = self.embedding([question])  # assuming it takes a list
        if not embed_query:
            logger.error("No embeddings returned for query.")
            query_projection = (np.array([]), np.array([]))
        else:
            query_embedding = np.array([d.embedding for d in embed_query])
            query_projection = self._project_embeddings(query_embedding, self.projector)

        # Prepare query dataframe
        if query_projection.size > 0:
            viz_query_df = pd.DataFrame(
                {
                    "x": [query_projection[0][0]],
                    "y": [query_projection[0][1]],
                    "document_cleaned": question,
                    "category": "Original Query",
                    "size": 5,
                }
            )
            visualization_df = pd.concat([viz_base_df, viz_query_df], axis=0)
        else:
            visualization_df = viz_base_df

        # Plot
        if visualization_df.empty:
            logger.error("Visualization DataFrame is empty. Skipping plot creation.")
            return None

        fig = self._plot_embeddings(visualization_df)
        logger.info("Citation plot created successfully.")
        return fig
