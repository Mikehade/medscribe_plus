import { useNavigate } from 'react-router-dom';
import '../styles/Dashboard.css';

function Home () {
    const navigate = useNavigate();
    return (
        <div>
            <h1>Welcome to the MedScribe Agent</h1>

            <center>
                
                <div className='link' onClick={() => navigate('/live_recording')}><b>Live Consultation</b></div>
                <div className='link' onClick={() => navigate('/upload_consultation')}><b>Pre-Recorded Consultation</b></div>
            </center>
            
        </div>
    );
};

export default Home;