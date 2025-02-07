import requests
from typing import Optional, Any
from concurrent.futures import Future
from threading import Thread
from qua_dashboards.utils.data_utils import serialise_data
from qua_dashboards.logging_config import logger


__all__ = ["DataDashboardClient"]


class DataDashboardClient:
    """
    Client for sending data to a Dash dashboard.

    If a URL is provided at initialization, that URL is used directly.
    Otherwise, the client starts a background thread to determine a reachable
    dashboard URL from a list of candidate URLs.
    """

    def __init__(self, url: Optional[str] = None):
        """
        Initialize the DataDashboardClient.

        Args:
            url (Optional[str]): Direct URL to the dashboard.
                If provided, automatic URL determination is skipped.
        """
        if url is not None:
            self.url: Optional[str] = url
            self._url_future: Optional[Future] = None
        else:
            self.url = None
            self._url_future = Future()
            Thread(target=self._determine_url, daemon=True).start()

    def _determine_url(self) -> None:
        """
        Determine the dashboard URL in a background thread.

        This method attempts to find a reachable URL from a list of candidate URLs.
        Once determined, it sets the result in the Future and assigns the URL.
        """
        url = self._get_dashboard_url()
        if self._url_future is not None:
            self._url_future.set_result(url)
        self.url = url

        if url is not None:
            logger.info(f"Data Dashboard - Using URL: {url}")

    @staticmethod
    def _get_dashboard_url() -> Optional[str]:
        """
        Check candidate URLs and return the first reachable dashboard URL.

        Returns:
            Optional[str]: A reachable dashboard URL if found; otherwise None
        """
        candidate_urls = [
            "http://localhost:8001/dashboards/data-dashboard",
            "http://localhost:8050/data-dashboard",
        ]
        for url in candidate_urls:
            try:
                response = requests.get(url, timeout=0.5)
                if response.ok:
                    return url
            except requests.RequestException as e:
                logger.debug(f"URL check failed for {url}: {e}")
        logger.warning(
            "Could not determine URL for Data Dashboard; Cannot send live data."
        )
        return None

    def send_data(self, data: Any) -> bool:
        """
        Serialize and send data to the dashboard.

        If the URL is still being determined, this method will wait (up to 5 seconds)
        for the URL before attempting to send data via a POST request to '{url}/update-data'.

        Args:
            data (Any): The data to be sent.

        Returns:
            bool: True if data was sent successfully; False otherwise.
        """
        # Wait for the URL to be determined if necessary.
        if self._url_future and not self._url_future.done():
            try:
                self.url = self._url_future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error determining Data Dashboard URL: {e}")
                self._url_future = None
                self.url = None
                return False

        if not self.url:
            logger.error("No Data Dashboard URL available; cannot send data.")
            return False

        # Serialize the data.
        serialised_data = serialise_data(data)

        try:
            response = requests.post(
                f"{self.url}/update-data", json=serialised_data, timeout=2
            )
            if response.ok:
                return True
            logger.error(
                f"Failed to send data to {self.url}, status code: {response.status_code}"
            )
            return False
        except requests.RequestException as e:
            logger.error(f"Exception occurred while sending data to {self.url}: {e}")
            return False
