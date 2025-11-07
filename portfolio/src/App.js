import './App.css';

// Task 3,6,9,12,15,18: Import the components here  
import Header from './components/Header';
import Body from './components/Body';
import About from './components/About';
import Projects from './components/Projects';
import Skills from './components/Skills';
import Footer from './components/Footer';

const App = () => {
  return (
    <div id="app" className="App">
      {/* Task 3,6,9,12,15,18: Use the imported components here */}
      <Header />
      <Body />
      <About />
      <Projects />
      <Skills />
      <Footer />
    </div>
  );
}

export default App;