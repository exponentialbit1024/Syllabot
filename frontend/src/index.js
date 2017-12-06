import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import Home from './Home';
import Auth from './utils/auth';
import history from './history';
import registerServiceWorker from './registerServiceWorker';
import Callback from './Components/Callback';
import { Router, Route,Switch } from 'react-router';
import { BrowserRouter } from 'react-router-dom'

const auth = new Auth();

const handleAuthentication = (nextState, replace) => {
  if (/access_token|id_token|error/.test(nextState.location.hash)) {
    auth.handleAuthentication();
  }
}

const Root = () => {
  return (
   <Router history={history} component={App}>
     <Switch>
        <Route exact path="/" render={(props) => <Home auth={auth} {...props} />}/>
        <Route exact path="/bot" render={(props) => <App auth={auth} {...props} />}/>
        <Route path="/callback" render={(props) => {
          handleAuthentication(props);
          return <Callback {...props} />
        }}/>
     </Switch>
    </Router>
  )
}

ReactDOM.render(<Root />, document.getElementById('root'));
registerServiceWorker();
