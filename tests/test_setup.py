"""
Phase 1 Setup Verification Tests

Purpose: Verify that the environment is correctly set up before proceeding
to implement core services.

What we test:
1. Configuration loads correctly
2. All required packages are installed
3. Settings have expected values
4. File structure is correct

Run this after:
- Installing requirements.txt
- Creating .env file (optional, uses defaults)

Usage:
    python tests/test_setup.py
    
Expected output:
    ✅ All setup tests passed!
"""

import sys
import os

def test_python_version():
    """
    Verify Python version is 3.11+
    
    Why 3.11+?
    - Type hints improvements
    - Better performance
    - pydantic v2 requires Python 3.7+
    """
    version = sys.version_info
    assert version.major == 3 and version.minor >= 11, \
        f"Python 3.11+ required, got {version.major}.{version.minor}"
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")


def test_imports():
    """
    Verify all required packages can be imported.
    
    Why test imports?
    - Catches missing dependencies early
    - Ensures requirements.txt was installed
    - Confirms no version conflicts
    """
    try:
        import fastapi
        print(f"✅ FastAPI installed: {fastapi.__version__}")
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install -r requirements.txt")
    
    try:
        import chromadb
        print(f"✅ ChromaDB installed: {chromadb.__version__}")
    except ImportError:
        raise ImportError("ChromaDB not installed. Run: pip install -r requirements.txt")
    
    try:
        import pydantic
        print(f"✅ Pydantic installed: {pydantic.__version__}")
    except ImportError:
        raise ImportError("Pydantic not installed. Run: pip install -r requirements.txt")
    
    try:
        import uvicorn
        print(f"✅ Uvicorn installed: {uvicorn.__version__}")
    except ImportError:
        raise ImportError("Uvicorn not installed. Run: pip install -r requirements.txt")
    
    print("✅ All required packages installed")


def test_config_loading():
    """
    Verify configuration loads correctly.
    
    Why test config?
    - Ensures settings class works
    - Validates default values
    - Catches typos in field names
    """
    from app.config import settings
    
    # Test basic settings
    assert isinstance(settings.APP_NAME, str), "APP_NAME must be string"
    assert isinstance(settings.CHUNK_SIZE, int), "CHUNK_SIZE must be int"
    assert isinstance(settings.CHUNK_OVERLAP, int), "CHUNK_OVERLAP must be int"
    
    # Test constraints
    assert settings.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
    assert settings.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE, \
        "CHUNK_OVERLAP must be less than CHUNK_SIZE"
    
    assert settings.DEFAULT_TOP_K > 0, "DEFAULT_TOP_K must be positive"
    assert settings.MAX_TOP_K >= settings.DEFAULT_TOP_K, \
        "MAX_TOP_K must be >= DEFAULT_TOP_K"
    
    print(f"✅ Configuration loaded successfully")
    print(f"   APP_NAME: {settings.APP_NAME}")
    print(f"   CHUNK_SIZE: {settings.CHUNK_SIZE}")
    print(f"   CHUNK_OVERLAP: {settings.CHUNK_OVERLAP}")
    print(f"   EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"   EMBEDDING_DIMENSION: {settings.EMBEDDING_DIMENSION}")


def test_project_structure():
    """
    Verify project directory structure exists.
    
    Why test structure?
    - Ensures all directories created
    - Catches missing folders before runtime errors
    - Validates project setup
    """
    required_dirs = [
        "app",
        "app/api",
        "app/services",
        "app/repositories",
        "app/models",
        "data",
        "tests"
    ]
    
    for dir_path in required_dirs:
        assert os.path.isdir(dir_path), f"Missing directory: {dir_path}"
    
    print("✅ Project structure verified")


def test_required_files():
    """
    Verify essential files exist.
    
    Why test files?
    - Ensures setup completed correctly
    - Catches missing configuration files
    - Validates git setup
    """
    required_files = [
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".gitignore",
        "README.md",
        "app/__init__.py",
        "app/main.py",
        "app/config.py",
        "app/models/schemas.py"
    ]
    
    for file_path in required_files:
        assert os.path.isfile(file_path), f"Missing file: {file_path}"
    
    print("✅ All required files present")


def test_fastapi_app():
    """
    Verify FastAPI app can be created.
    
    Why test app?
    - Ensures no import errors in main.py
    - Validates FastAPI configuration
    - Catches circular imports
    """
    from app.main import app
    
    assert app is not None, "FastAPI app not created"
    assert hasattr(app, 'routes'), "FastAPI app missing routes"
    
    # Check essential endpoints exist
    routes = [route.path for route in app.routes]
    assert "/" in routes, "Root endpoint missing"
    assert "/health" in routes, "Health endpoint missing"
    
    print("✅ FastAPI application initialized")
    print(f"   Available routes: {len(routes)}")


def test_data_directory():
    """
    Verify data directory exists and is writable.
    
    Why test data directory?
    - Chroma needs write permissions
    - Prevents runtime errors during indexing
    - Ensures persistence will work
    """
    from app.config import settings
    
    data_dir = settings.CHROMA_PERSIST_DIRECTORY
    
    # Create directory if doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Test write permissions
    test_file = os.path.join(data_dir, ".test_write")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Data directory writable: {data_dir}")
    except Exception as e:
        raise PermissionError(f"Cannot write to data directory: {e}")


def run_all_tests():
    """
    Run all setup verification tests.
    
    Order matters:
    1. Python version (prerequisite for everything)
    2. Package imports (need packages installed)
    3. Configuration (needs packages)
    4. Structure (needs config to reference paths)
    5. FastAPI app (needs structure)
    6. Data directory (final check)
    """
    print("=" * 60)
    print("🔍 Running Phase 1 Setup Verification Tests")
    print("=" * 60)
    print()
    
    try:
        test_python_version()
        print()
        
        test_imports()
        print()
        
        test_config_loading()
        print()
        
        test_project_structure()
        print()
        
        test_required_files()
        print()
        
        test_fastapi_app()
        print()
        
        test_data_directory()
        print()
        
        print("=" * 60)
        print("🎉 All Phase 1 Setup Tests Passed!")
        print("=" * 60)
        print()
        print("✅ Environment is ready for Phase 2 (Core Services)")
        print()
        print("Next steps:")
        print("1. Create .env file (copy from .env.example)")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Test API: uvicorn app.main:app --reload")
        print("4. Proceed to Phase 2: Implement core services")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
