class FinancialRiskPlatformError(Exception):
    pass
class TickerResolutionError(FinancialRiskPlatformError):
    pass
class DataDownloadError(FinancialRiskPlatformError):
    pass
class DataPreprocessingError(FinancialRiskPlatformError):
    pass
class FeatureEngineeringError(FinancialRiskPlatformError):
    pass
class InferenceDataError(FinancialRiskPlatformError):
    pass
class ModelLoadingError(FinancialRiskPlatformError):
    pass
class PredictionError(FinancialRiskPlatformError):
    pass
class SHAPExplanationError(FinancialRiskPlatformError):
    pass
class LLMContextError(FinancialRiskPlatformError):
    pass
class LLMReportError(FinancialRiskPlatformError):
    pass
class InferencePipelineError(FinancialRiskPlatformError):
    pass