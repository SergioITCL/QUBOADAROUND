class LayerType:
    @staticmethod
    def is_fully_connected(layerCode: str):
        return "FULLY_CONN" in layerCode.upper()

    @staticmethod
    def is_tanh(layerCode: str):
        return "TANH" == layerCode.upper()

    @staticmethod
    def is_tanhLUT(layerCode: str):
        return "TANHLUT" == layerCode.upper()

    @staticmethod
    def is_logistic(layerCode: str):
        return "LOGISTIC" in layerCode.upper()

    @staticmethod
    def is_softmax(layerCode: str):
        return "SOFTMAX" in layerCode.upper()

    @staticmethod
    def is_quantize(layerCode: str):
        return "QUANTIZE" == layerCode.upper()

    @staticmethod
    def is_dequantize(layerCode: str):
        return "DEQUANTIZE" in layerCode.upper()

    @staticmethod
    def is_sigmoidLUT(layerCode: str):
        return "SIGMOIDLUT" in layerCode.upper()

    @staticmethod
    def is_relu(layerCode: str):
        return "RELU" == layerCode.upper()

    @staticmethod
    def is_LSTM(layerCode: str):
        return "LSTM" == layerCode.upper()
