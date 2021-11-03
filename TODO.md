<p align="center">
    <img src="./assets/BINGO_LOGO-cor.png" width=20%
        alt = "Release Status">
</p>

# Estudo do Feixe

- Criar script `make_dataset.py` que recebe como entrada um arquivo xlsx do tipo fornecido pelos relatórios do LIT.
    - Detecta número `dataset_num` e nome das planilhas `dataset_name`
    - Gera `dataset_num` planilhas no formato `csv` com codificação UTF-8 e nome padronizado, com os dados de um dataframe que contem colunas `frequencia`, `amplitude` e `fase` em formato tidy data (uma linha por observação).
    - O nome padronizado segue:
    ```bash
    <nome-do-instrumento>_dataset_name_num_dataset_num.csv
    ```
    - Arquivos de dados são armazenados em /data/processed
    - script armazenada em /src/data/

# package: RADIOTELESCOPE

## module: GRASP

### CLASS Coordinate

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **origin**
     - `@property ` **euler_angles**
     - `@property ` **base_frame**
     - SetRotationMatrix(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)

### CLASS Surface

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **frame**
     - `@property ` **centre**
     - `@property ` **base_frame**
     - `@property ` **geometry**
     - SetRotationMatrix(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)
     - deform(self,*args, *kwargs)

### CLASS Rim

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **frame**
     - `@property ` **centre**
     - `@property ` **base_frame**
     - `@property ` **geometry**
     - SetRotationMatrix(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)
     - deform(self,*args, *kwargs)

### CLASS Reflector

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **frame**
     - `@property ` **centre**
     - `@property ` **base_frame**
     - `@property ` **geometry**
     - SetRotationMatrix(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)
     - deform(self,*args, *kwargs)

### CLASS Feed

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **frame**
     - `@property ` **taper**
     - `@property ` **taper_angle**
     - `@property ` **frequency**
     - SetRotationMatrix(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)

### CLASS Current (FAZENDO)

### CLASS Field (FAZENDO)

### CLASS Analysis (FAZENDO)

### CLASS Instrument  (FAZENDO)

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - AddReflector(self,*args, *kwargs)
     - AddFeed(self,*args, *kwargs)
     - AddCurrent(self,*args, *kwargs)
     - to_dict(self,*args, *kwargs)
     - from_dict(self,*args, *kwargs)
     - AddCut(self,*args, *kwargs)
     - RunCut(self,*args, *kwargs)
     - to_TOR(self,*args, *kwargs)
     - from_TOR(self,*args, *kwargs)
     - to_CUT(self,*args, *kwargs)
     - from_CUT(self,*args, *kwargs)
     - to_GRD(self,*args, *kwargs)
     - deform(self,*args, *kwargs)

## module GEOMETRY

### CLASS Coordinate

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - Draw(self,*args, *kwargs)

### CLASS Point

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - Draw(self,*args, *kwargs)

### CLASS Line

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - GetLength(self,*args, *kwargs)
     - GetTangent(self,*args, *kwargs)
     - GetNormal(self,*args, *kwargs)
     - GetBoundaries(self,*args, *kwargs)
     - Draw(self,*args, *kwargs)

### CLASS Plane

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - GetLength(self,*args, *kwargs)
     - GetTangent(self,*args, *kwargs)
     - GetNormal(self,*args, *kwargs)
     - GetBoundaries(self,*args, *kwargs)
     - Draw(self,*args, *kwargs)

### CLASS Paraboloyd

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - GetLength(self,*args, *kwargs)
     - GetTangent(self,*args, *kwargs)
     - GetNormal(self,*args, *kwargs)
     - GetBoundaries(self,*args, *kwargs)
     - Draw(self,*args, *kwargs)

### CLASS Hyperboloid

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - GetLength(self,*args, *kwargs)
     - GetTangent(self,*args, *kwargs)
     - GetNormal(self,*args, *kwargs)
     - GetBoundaries(self,*args, *kwargs)
     - Draw(self,*args, *kwargs)

### CLASS CylindricalCut

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **dimension**
     - GetLength(self,*args, *kwargs)
     - GetTangent(self,*args, *kwargs)
     - GetNormal(self,*args, *kwargs)
     - GetBoundaries(self,*args, *kwargs)
     - Draw(self,*args, *kwargs)

## module: ANTENNA

### CLASS Antenna

- **Methods**
     - __init__(self,*args, *kwargs)
     - `@property ` **name**
     - `@property ` **SpilloverEff**
     - `@property ` **OhmicEff**
     - `@property ` **PolarizationEff**
     - `@property ` **PhaseEff**
     - `@property ` **Coma**
     - `@property ` **Astimagtism**
     - `@property ` **Blockage**
     - `@property ` **Directivity**
     - `@property ` **Front2BackRatio**
     - GetPattern(self,*args, *kwargs)
     - LoadPattern(self,*args, *kwargs)
     - DrawPattern(self,*args, *kwargs)
