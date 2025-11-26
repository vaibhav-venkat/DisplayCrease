"""Aggregate callback registration helpers."""

from . import data_upload, ga_callbacks, results_upload, tab_content, visualizations


def register_callbacks(app):
	"""Register every callback module with the supplied Dash app."""
	tab_content.register(app)
	data_upload.register(app)
	results_upload.register(app)
	visualizations.register(app)
	ga_callbacks.register(app)
