// Task 5:  Create the Body component here
// Importing the GitHub and LinkedIn icons rom react-icons/fa and an avatar image from the assets folder
import { FaGithub, FaLinkedin } from "react-icons/fa";
import Avatar from '../assets/Software.png';

// Defining the Body functional component
const Body = () => {
    return (
        // Main container for the body section
        <div id="body" className="body">
            <div className="body-container">

               {/* Profile section containing the avatar, name, title, and social icons */}
                <div className="body-profile">
                    <img className="body-img" alt='avatar' src={Avatar} />

                    {/* Container for user details */}
                    <div className="body-content">
                        <div className="body-headline">Srikala Gangi Reddy</div>
                        <div className="body-text">Software Engineering Manager</div>
                    </div>

                    {/* Social media icons */}
                    <div className="body-icons">
                        <a href="https://github.com/srikala-g/AI-Portfolio" target="_blank" rel="noreferrer" className="icon-link"><i><FaGithub /></i> </a>
                        <a href="https://www.linkedin.com/in/srikala-gangi-reddy/" target="_blank" rel="noreferrer" className="icon-link"><i><FaLinkedin /></i></a>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Body;
