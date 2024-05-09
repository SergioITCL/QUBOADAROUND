# ITCL Quantization Toolkit


## Instalación

Requiere Python 3.10 y [Poetry](https://python-poetry.org/)

No es recomendable el uso de entornos virtuales con Poetry, por lo que se desaconseja el uso de VENV o Conda como el entorno activo por defecto. Si se usa Conda, el entorno Base deberá usar Python 3.10.

Poetry generará un .venv para cada proyecto con las dependencias individuales.

Es importante comprobar mediante ```python -V``` y ```pip -V``` que la versión de python y pip actual son las correspondientes a Python 3.10.

### Instalar Poetry.

Opción directa para entornos sin conflictos de dependencias:
```bash
pip install poetry
```
Si el entorno incluye alguna dependencia que pueda causar algún conflicto con Poetry,
 es recomendable instalar Poetry mediante el 
 [script de instalación](https://python-poetry.org/docs/#installation)

### Ejecución

Sobre un proyecto cualquiera, por ejemplo ```./quantizer```, instalaremos 
el proyecto mediante el siguiente comando.

```bash
cd ./quantizer
poetry install
```
Este paso deberá generar un entorno bajo el directorio ```.venv``` 
con el siguiente mensaje 
```Creating virtualenv itcl-quantizer in .../Quantize-Inference/quantizer/.venv```

Para ejecutar uno de los cuantizadores usaremos el comando:

```bash
poetry run python ./quantize_mnist.py
```

### Otros Comandos

Podemos añadir un nuevo paquete al proyecto mediante:

```bash
poetry add pytest@latest --dev
```

donde ```@latest``` hace referencia a la última versión disponible y ```--dev``` 
a que sea una dependencia de desarrollo. 
Estas dependencias no serán incluidas en el despliegue.

Otro comando interesante es ```poetry shell```, que nos abre un terminal con el entorno del proyecto activado.

Se recomienda consultar la [documentación oficial](https://python-poetry.org/docs/cli/) 
para detallar el resto de comandos, como ```update```, ```remove```, ```init```, etc.