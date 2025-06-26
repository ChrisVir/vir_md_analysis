# Suggested Module Reorganization for vir_md_analysis

## Current Issues
1. **`structural_features.py` is too large** (688 lines) with mixed responsibilities
2. **Missing separation** between feature computation and statistical analysis  
3. **Feature registration/mapping** is hardcoded in `feature_extraction.py`
4. **No clear separation** between MDTraj and PyTraj implementations
5. **Utils scattered** across different files

## Proposed New Structure

```
vir_md_analysis/
├── features/
│   ├── __init__.py
│   ├── core/                    # Core feature computation
│   │   ├── __init__.py
│   │   ├── mdtraj_features.py   # MDTraj-specific implementations
│   │   ├── pytraj_features.py   # PyTraj-specific implementations
│   │   └── base.py              # Base classes/interfaces
│   ├── extractors/              # High-level feature extraction
│   │   ├── __init__.py
│   │   ├── extractor.py         # Main extraction orchestrator
│   │   ├── registry.py          # Feature function registry
│   │   └── config.py            # Configuration management
│   ├── analysis/                # Statistical analysis
│   │   ├── __init__.py
│   │   ├── statistics.py        # Statistical computations
│   │   └── regions.py           # Region-based analysis
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── loaders.py           # File loaders (existing)
│   │   ├── topology_utils.py    # Topology utilities (existing)
│   │   └── constants.py         # Constants (max_aa_sasa, etc.)
│   └── io/                      # Input/Output
│       ├── __init__.py
│       ├── writers.py           # Result writers
│       └── validators.py        # Input validation
```

## Migration Plan

### Phase 1: Create new structure and move utilities
1. Create new directory structure
2. Move constants to `utils/constants.py`
3. Keep existing `loaders.py` and `topology_utils.py` in `utils/`

### Phase 2: Split structural_features.py by feature type
1. Create `core/mdtraj_features.py` with:
   - SASA calculations (`compute_sasa_per_frame`)
   - Hydrogen bond analysis (`identify_hydrogen_bonds`)
   - RMSD/RMSF calculations (`compute_rmsd_on_specific_regions`, `compute_rmsf_on_specific_regions`)
   - Radius of gyration (`compute_radius_of_gyration_by_region`)
   - Contacts (`compute_contacts_on_specific_regions`)
   - Topology utilities (`get_topology_dataframe`, etc.)

2. Create `core/pytraj_features.py` with:
   - S2 parameter calculation (`calculate_s2_parameter`)
   - N-H bond identification (`identify_nh_bonds`)
   - Chain utilities (`select_chains`, etc.)

3. Create `analysis/statistics.py` with:
   - `compute_stats_from_series`
   - Statistical helper functions

### Phase 3: Create feature registry system
1. Create `extractors/registry.py` for feature function registration
2. Create `extractors/config.py` for configuration management
3. Move hardcoded feature maps from `feature_extraction.py`

### Phase 4: Refactor main extraction logic
1. Update `extractors/extractor.py` (renamed from `feature_extraction.py`)
2. Use registry system instead of hardcoded mappings
3. Improve error handling and validation

## Benefits

### 🎯 **Improved Maintainability**
- **Single Responsibility**: Each module has a clear, focused purpose
- **Easier Testing**: Smaller, focused modules are easier to unit test
- **Better Documentation**: Each module can have specific documentation

### 🔧 **Enhanced Extensibility**
- **Feature Registry**: Easy to add new features without modifying core code
- **Plugin Architecture**: New feature types can be added as plugins
- **Configuration Management**: Features can be configured via files/parameters

### 📊 **Better Separation of Concerns**
- **Computation vs Analysis**: Clear separation between feature computation and statistical analysis
- **Library Abstraction**: MDTraj and PyTraj implementations are clearly separated
- **IO Handling**: Input/output logic is isolated

### 🚀 **Performance & Scalability**
- **Selective Imports**: Only import needed modules
- **Lazy Loading**: Features can be loaded on-demand
- **Caching**: Statistical computations can be cached

## Implementation Example

### New Feature Registry (`extractors/registry.py`)
```python
from typing import Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    name: str
    function: Callable
    library: str  # 'mdtraj' or 'pytraj'
    description: str
    default_params: Dict[str, Any]

class FeatureRegistry:
    def __init__(self):
        self._features = {}
    
    def register(self, config: FeatureConfig):
        self._features[config.name] = config
    
    def get_mdtraj_features(self) -> Dict[str, Callable]:
        return {name: config.function 
                for name, config in self._features.items() 
                if config.library == 'mdtraj'}
```

### Configuration-driven extraction (`extractors/config.py`)
```python
from pathlib import Path
import yaml

class ExtractionConfig:
    def __init__(self, config_path: Path = None):
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_defaults()
    
    def load_defaults(self):
        self.features = {
            'rmsd': {'enabled': True, 'params': {}},
            'rmsf': {'enabled': True, 'params': {}},
            'sasa': {'enabled': True, 'params': {'relative': True}},
            # ... etc
        }
```

## Backward Compatibility

- Keep existing imports working with `__init__.py` re-exports
- Provide deprecation warnings for direct imports
- Gradual migration path for existing code

## Testing Strategy

- **Unit tests** for each feature module
- **Integration tests** for the extraction pipeline
- **Performance benchmarks** to ensure no regression
- **API compatibility tests** for backward compatibility
