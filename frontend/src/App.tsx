import Home from "./pages/Home"
import StoryPredictorPage from "./pages/chat-1"
import LiveRecording  from "./pages/LiveRecording"	
import UploadConsultation  from "./pages/UploadCOnsultation"


import { BrowserRouter as Router, Routes, Route } from "react-router-dom"

function App() {

	return (
		<Router>
			<Routes>
				<Route path="/" element={<Home />} />
				<Route path="/virality_evaluator_with_rewrite" element={<StoryPredictorPage />} />
				<Route path="/live_recording" element={<LiveRecording />} />
				<Route path="/upload_consultation" element={<UploadConsultation />} />

			</Routes>
		</Router>
	)
}

export default App