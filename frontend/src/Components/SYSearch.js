import React, { Component } from 'react';
import PropTypes from 'prop-types';
import ChatBot, { Loading } from 'react-simple-chatbot';
import { getAns } from '../utils/apiUtil';

class SYSearch extends Component {
  constructor(props) {
    super(props);

    this.state = {
      loading: true,
      result: '',
      trigger: false,
    };

    this.triggetNext = this.triggetNext.bind(this);
  }

  componentWillMount() {
    const self = this;
    const { steps } = this.props;
    const search = steps.search.value;
    getAns(search).then((result) => {
    console.log(result);
    this.setState({
      result: result.response,
      loading: false
    });
  });
  }

  triggetNext() {
    this.setState({ trigger: true }, () => {
      this.props.triggerNextStep();
    });
  }

  render() {
    const { trigger, loading, result } = this.state;

    return (
      <div className="dbpedia">
        { loading ? <Loading /> : result }
        {
          !loading &&
          <div
            style={{
              textAlign: 'center',
              marginTop: 20,
            }}
          >
            {
              !trigger &&
              <button
                onClick={() => this.triggetNext()}
              >
                Search Again
              </button>
            }
          </div>
        }
      </div>
    );
  }
}

SYSearch.propTypes = {
  steps: PropTypes.object,
  triggerNextStep: PropTypes.func,
};

SYSearch.defaultProps = {
  steps: undefined,
  triggerNextStep: undefined,
};

export default SYSearch;
