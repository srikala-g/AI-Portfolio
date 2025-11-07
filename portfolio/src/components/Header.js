// Task 2: Implement the Header component here

// Define a functional component named Header
const Header = () => {
    return (
        // Main container for the header section  containing a container for a header logo and navigation items
        <div className="header">
                <div className="header-logo">
                    <a href="#app"><h3 className="header-name">Srikala Gangi Reddy</h3></a>
                </div>
                <div className="header-container">
                    <div className="nav-item"><a href="#about">About</a></div>
                    <div className="nav-item"><a href="#projects">Projects</a></div>
                    <div className="nav-item"><a href="#skills">Skills</a></div>
                </div>
        </div>
    )
}

// Export the Header component for use in other parts of the application
export default Header;