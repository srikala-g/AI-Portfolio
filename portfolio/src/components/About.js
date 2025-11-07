// Task 8: Create your About component here
import IMG from '../assets/Detective.png';

const About = () => {
    return (
        <div id="about" className="about">
            <h1 className="about-heading">About Me</h1>
            <div className="about-info">
                <p className="about-desc">I’m Srikala — a Software Engineering Manager and Visual Artist passionate about uniting technology and creativity. With over a decade of experience delivering enterprise-scale solutions at Fidelity, Microsoft, and Intel, I now focus on leading AI-driven projects that blend innovation, design thinking, and conscious leadership to create purposeful digital experiences.</p>
                <div className="about-img">
                    <div className="about-img-wrapper">
                        <img src={IMG} alt="Detective" />
                    </div>
                </div>
            </div>
        </div>
    )
}

export default About;