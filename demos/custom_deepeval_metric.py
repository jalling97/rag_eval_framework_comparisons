from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import asyncio

# Inherit BaseMetric
class LatencyMetric(BaseMetric):
    # This metric by default checks if the latency is greater than 10 seconds
    def __init__(
            self, 
            max_seconds: int = 10,
            async_mode: bool = True,
            threshold = 1
        ):

        self.max_seconds = max_seconds
        self.threshold = threshold
        self.async_mode = async_mode

    def measure(self, test_case: LLMTestCase):
        # Set self.success and self.score in the "measure" method
        self.success = test_case.additional_metadata['latency'] <= self.max_seconds
        if self.success:
            self.score = 1
            self.reason = "Latency was below the acceptable limit of {} seconds".format(self.threshold)
        else:
            self.score = 0
            self.reason = "Too slow!"
        
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.measure, test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"