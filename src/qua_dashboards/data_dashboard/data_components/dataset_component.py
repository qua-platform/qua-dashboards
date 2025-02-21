import xarray as xr
from typing import Type, Optional, TypeVar
from dash.development.base_component import Component
import dash_bootstrap_components as dbc
from dash import html
from qua_dashboards.data_dashboard.data_components.base_data_component import (
    BaseDataComponent,
)
from qua_dashboards.data_dashboard.data_components.data_array_component import (
    DataArrayComponent,
)
from qua_dashboards.logging_config import logger

T = TypeVar("T", bound=Component)


class DatasetComponent(BaseDataComponent):
    @classmethod
    def can_handle(cls, value: any) -> bool:
        # Allow any Dataset regardless of DataArray ndim.
        return isinstance(value, xr.Dataset)

    @classmethod
    def create_component(
        cls,
        label: str,
        value: xr.Dataset,
        existing_component: Optional[Component] = None,
        root_component_class: Type[T] = html.Div,
    ) -> T:
        """
        Create or update a component for visualizing an xarray Dataset with
        multiple DataArrays. Each DataArray is rendered using the
        DataArrayComponent, which handles ND arrays using sliders when needed.
        """
        if not cls._validate_existing_component(
            existing_component, value, root_component_class
        ):
            logger.info("Creating new dataset component (%s)", label)
            root_component = cls.create_collapsible_root_component(
                label, root_component_class, "xarray_dataset_component"
            )
            collapse_comp = root_component.children[1]
            data_array_cols = []
            for name, da in value.data_vars.items():
                sub_label = f"{label}/{name}"
                sub_comp = DataArrayComponent.create_component(
                    sub_label, da, None, html.Div
                )
                data_array_cols.append(dbc.Col(sub_comp))
            collapse_comp.children = dbc.Row(data_array_cols, style={"display": "flex"})
        else:
            logger.info("Using existing dataset component (%s)", label)
            root_component = existing_component
            collapse_comp = root_component.children[1]
            # Update each DataArray sub-component.
            for col, (name, da) in zip(
                collapse_comp.children.children, value.data_vars.items()
            ):
                sub_label = f"{label}/{name}"
                updated_comp = DataArrayComponent.create_component(
                    sub_label, da, col.children, html.Div
                )
                col.children = updated_comp.children
        return root_component

    @staticmethod
    def _validate_existing_component(
        component: Component, value: xr.Dataset, root_component_class: Type[T]
    ) -> bool:
        if not isinstance(component, root_component_class):
            return False
        if getattr(component, "data-class", None) != "xarray_dataset_component":
            return False
        collapse_comp = component.children[1]
        try:
            data_array_cols = collapse_comp.children.children
            labels = [col.children.children[0].children for col in data_array_cols]
            # Expect labels in the form "label/data_array_name"
            html_names = [lbl.split("/", 1)[1] for lbl in labels]
            xarray_names = list(value.data_vars.keys())
            return html_names == xarray_names
        except Exception as e:
            logger.error("Validation failed: %s", e)
            return False
