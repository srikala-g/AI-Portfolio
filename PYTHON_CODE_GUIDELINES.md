# PYTHON_CODE_GUIDELINES.md
**Python Coding Standards & Best Practices**

---

## 1. General Principles
- Write **clean, readable, and maintainable** code.  
- Prefer **clarity over cleverness**.  
- Apply **SOLID** principles and **DRY (Don’t Repeat Yourself)**.  
- Use **meaningful names** for variables, functions, and classes.  
- Keep functions and classes **focused** on a single responsibility.  

---

## 2. Style & Naming Conventions
- Follow **PEP 8** coding style.  
- **snake_case** → variables & functions (`calculate_area`)  
- **PascalCase** → classes (`CircleCalculator`)  
- **UPPER_CASE** → constants (`PI = 3.14159`)  
- Indentation: **4 spaces** (never tabs).  
- Limit lines to **≤ 88 characters** (Black formatter default).  

---

## 3. Documentation
- Write **docstrings** (PEP 257) for modules, classes, and functions.  
- Example:
  ```python
  def calculate_area(radius: float) -> float:
      """Calculate the area of a circle.

      Args:
          radius (float): Circle radius.

      Returns:
          float: The circle's area.
      """
      return 3.14159 * radius ** 2
  ```

---

## 4. Error Handling
- Catch **specific exceptions** (not bare `except`).  
- Use Python’s `logging` module instead of `print()` for errors.  
- Raise custom exceptions for domain-specific errors.  

---

## 5. Testing
- Use **pytest** for testing.  
- Follow **AAA (Arrange, Act, Assert)** pattern.  
- Example:
  ```python
  def test_calculate_area():
      # Arrange
      radius = 2.0
      # Act
      result = calculate_area(radius)
      # Assert
      assert result == 12.56636
  ```

---

## 6. Type Hints & Linting
- Always use **type hints**.  
- Run `mypy` or `pyright` for type checking.  
- Use `black` + `isort` + `flake8` for formatting and linting.  

---

## 7. Project Structure
```
project/
│── src/
│   ├── module1.py
│   ├── module2.py
│── tests/
│   ├── test_module1.py
│── requirements.txt / pyproject.toml
│── README.md
│── PYTHON_CODE_GUIDELINES.md
```

---

## 8. Security
- Do not hardcode secrets → use environment variables (`os.getenv`).  
- Sanitize inputs when dealing with user data.  
- Avoid using `eval` or other unsafe functions.  

---

## 9. Performance
- Use built-in data structures efficiently (`dict`, `set`, `list`).  
- Use **list comprehensions** instead of loops when possible.  
- Use **asyncio** or multiprocessing for I/O or CPU-bound tasks.  

---

## 10. Code Review Checklist
- ✅ Code follows PEP 8.  
- ✅ Functions are small, readable, and type-annotated.  
- ✅ Proper error handling and logging.  
- ✅ Unit tests cover core logic.  
- ✅ No secrets or sensitive data in code.  
